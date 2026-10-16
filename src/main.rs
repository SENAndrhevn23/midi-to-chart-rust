use eframe::egui;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode { Single, Dual }
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mapping { AccurateRanges, Adaptive, SmartBalance }
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Style { Classic, Stream, ChordFriendly, Midi2PsychMapping }
impl Style {
    fn label(self) -> &'static str {
        match self {
            Self::Classic => "Classic",
            Self::Stream => "Stream",
            Self::ChordFriendly => "Chord-friendly",
            Self::Midi2PsychMapping => "Midi2Psych Mapping",
        }
    }
}
#[derive(Clone)]
struct Note {
    ms: f64,
    end_ms: f64,
    pitch: u8,
    channel: u8,
    track: u32,
    player_two: bool,
}
#[derive(Default)]
struct MidiInfo { notes: Vec<Note>, note_count: usize, bpm: f64, tracks: usize, duration: f64 }

fn stream_skip<R: Seek>(reader: &mut R, bytes: u64) -> Result<(), String> {
    let offset = i64::try_from(bytes).map_err(|_| "MIDI chunk is too large to seek on this platform.".to_string())?;
    reader.seek(SeekFrom::Current(offset))
        .map(|_| ())
        .map_err(|e| format!("Couldn't skip MIDI chunk data: {e}"))
}

fn read_track_byte<R: Read>(reader: &mut R, remaining: &mut u64) -> Result<u8, String> {
    if *remaining == 0 {
        return Err("Malformed MIDI: a track event extends past its declared track length.".into());
    }
    let mut byte = [0u8; 1];
    reader.read_exact(&mut byte)
        .map_err(|_| "Malformed MIDI: unexpected end of file inside a track.".to_string())?;
    *remaining -= 1;
    Ok(byte[0])
}

fn read_track_data<R: Read>(reader: &mut R, remaining: &mut u64, out: &mut [u8]) -> Result<(), String> {
    if out.len() as u64 > *remaining {
        return Err("Malformed MIDI: an event's data exceeds its track length.".into());
    }
    reader.read_exact(out)
        .map_err(|_| "Malformed MIDI: unexpected end of file inside a track.".to_string())?;
    *remaining -= out.len() as u64;
    Ok(())
}

fn skip_track_bytes<R: Read>(reader: &mut R, remaining: &mut u64, mut bytes: u64) -> Result<(), String> {
    if bytes > *remaining {
        return Err("Malformed MIDI: an event's data exceeds its track length.".into());
    }
    let mut scratch = [0u8; 16 * 1024];
    while bytes > 0 {
        let take = bytes.min(scratch.len() as u64) as usize;
        read_track_data(reader, remaining, &mut scratch[..take])?;
        bytes -= take as u64;
    }
    Ok(())
}

fn read_track_vlq<R: Read>(reader: &mut R, remaining: &mut u64) -> Result<u32, String> {
    let mut value = 0u32;
    for _ in 0..4 {
        let byte = read_track_byte(reader, remaining)?;
        value = (value << 7) | u32::from(byte & 0x7f);
        if byte & 0x80 == 0 { return Ok(value); }
    }
    Err("Malformed MIDI: variable-length quantity exceeds four bytes.".into())
}

fn read_track_data_byte<R: Read>(reader: &mut R, remaining: &mut u64) -> Result<u8, String> {
    let byte = read_track_byte(reader, remaining)?;
    if byte & 0x80 != 0 {
        return Err("Malformed MIDI: expected a data byte but found a status byte.".into());
    }
    Ok(byte)
}

fn push_parsed_note(notes: &mut Vec<Note>, note: Note) -> Result<(), String> {
    // Try to reserve before growing so memory pressure becomes a readable error
    // rather than an allocator panic while parsing a very large file.
    if notes.len() == notes.capacity() {
        notes.try_reserve(16_384).map_err(|_| {
            "Not enough memory to retain all MIDI notes. This file has too many notes for the available RAM; try splitting the MIDI into smaller files or use a machine with more memory.".to_string()
        })?;
    }
    notes.push(note);
    Ok(())
}

fn push_tempo_event(tempo_events: &mut Vec<(u64, u32)>, event: (u64, u32)) -> Result<(), String> {
    if tempo_events.len() == tempo_events.capacity() {
        tempo_events.try_reserve(32).map_err(|_| {
            "Not enough memory to retain the MIDI tempo map. The file may contain an unreasonable number of tempo events.".to_string()
        })?;
    }
    tempo_events.push(event);
    Ok(())
}

fn push_active_start(starts: &mut VecDeque<u64>, tick: u64) -> Result<(), String> {
    if starts.len() == starts.capacity() {
        starts.try_reserve(64).map_err(|_| {
            "Not enough memory to track overlapping notes. Try splitting the MIDI into smaller files or use a machine with more RAM.".to_string()
        })?;
    }
    starts.push_back(tick);
    Ok(())
}

fn parse_midi_track<R: Read>(
    reader: &mut R,
    track_len: u64,
    track_index: u32,
    notes: &mut Vec<Note>,
    tempo_events: &mut Vec<(u64, u32)>,
) -> Result<(), String> {
    let mut remaining = track_len;
    let mut tick = 0u64;
    let mut running_status: Option<u8> = None;
    let mut active: BTreeMap<(u8, u8), VecDeque<u64>> = BTreeMap::new();

    while remaining > 0 {
        let delta = read_track_vlq(reader, &mut remaining)? as u64;
        tick = tick.checked_add(delta).ok_or_else(|| "MIDI track tick position overflowed a 64-bit counter.".to_string())?;
        let first = read_track_byte(reader, &mut remaining)?;
        let (status, first_data) = if first & 0x80 != 0 {
            if first < 0xf0 {
                running_status = Some(first);
            } else if first != 0xff {
                // System-common and SysEx messages cancel channel running status.
                running_status = None;
            }
            (first, None)
        } else {
            let status = running_status.ok_or_else(|| {
                "Malformed MIDI: running-status data appeared before a channel status byte.".to_string()
            })?;
            (status, Some(first))
        };

        match status {
            0x80..=0xef => {
                let command = status & 0xf0;
                let channel = status & 0x0f;
                let data1 = match first_data {
                    Some(value) => value,
                    None => read_track_data_byte(reader, &mut remaining)?,
                };
                let has_second = !matches!(command, 0xc0 | 0xd0);
                let data2 = if has_second { read_track_data_byte(reader, &mut remaining)? } else { 0 };
                match command {
                    0x90 if data2 > 0 => {
                        let starts = active.entry((channel, data1)).or_default();
                        push_active_start(starts, tick)?;
                    }
                    0x80 | 0x90 => {
                        if let Some(starts) = active.get_mut(&(channel, data1)) {
                            if let Some(start) = starts.pop_front() {
                                push_parsed_note(notes, Note {
                                    ms: start as f64, end_ms: tick as f64, pitch: data1,
                                    channel, track: track_index, player_two: false,
                                })?;
                            }
                        }
                    }
                    _ => {}
                }
            }
            0xff => {
                // Meta events are ignored except tempo. Running status remains available
                // across meta events, as commonly encoded by Standard MIDI Files.
                let meta_type = read_track_byte(reader, &mut remaining)?;
                let payload_len = read_track_vlq(reader, &mut remaining)? as u64;
                if meta_type == 0x51 && payload_len == 3 {
                    let mut bytes = [0u8; 3];
                    read_track_data(reader, &mut remaining, &mut bytes)?;
                    let tempo = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
                    if tempo > 0 { push_tempo_event(tempo_events, (tick, tempo))?; }
                } else {
                    skip_track_bytes(reader, &mut remaining, payload_len)?;
                }
                if meta_type == 0x2f {
                    // End-of-track marks the logical end. Skip any malformed padding
                    // so the next chunk starts at the correct byte offset.
                    let trailing_bytes = remaining;
                    skip_track_bytes(reader, &mut remaining, trailing_bytes)?;
                    break;
                }
            }
            0xf0 | 0xf7 => {
                let payload_len = read_track_vlq(reader, &mut remaining)? as u64;
                skip_track_bytes(reader, &mut remaining, payload_len)?;
                running_status = None;
            }
            0xf1 | 0xf3 => {
                let _ = read_track_data_byte(reader, &mut remaining)?;
            }
            0xf2 => {
                let _ = read_track_data_byte(reader, &mut remaining)?;
                let _ = read_track_data_byte(reader, &mut remaining)?;
            }
            // System real-time and tune-request events have no payload in a MIDI file.
            0xf6 | 0xf8..=0xfe => {}
            _ => return Err(format!("Unsupported or malformed MIDI status byte 0x{status:02X}.")),
        }
    }

    for ((channel, pitch), starts) in active {
        for start in starts {
            push_parsed_note(notes, Note {
                ms: start as f64, end_ms: tick as f64, pitch,
                channel, track: track_index, player_two: false,
            })?;
        }
    }
    Ok(())
}

fn parse_midi(path: &Path) -> Result<MidiInfo, String> {
    // Stream events from disk instead of allocating both the entire file and a
    // parsed event tree. This keeps the input-size cost close to the final note list.
    let file = File::open(path).map_err(|e| format!("Couldn't open MIDI: {e}"))?;
    let file_size = file.metadata().map_err(|e| format!("Couldn't inspect MIDI file: {e}"))?.len();
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    let mut chunk_header = [0u8; 8];
    reader.read_exact(&mut chunk_header).map_err(|_| "This file is too short to be a Standard MIDI File.".to_string())?;
    if &chunk_header[..4] != b"MThd" {
        return Err("This file doesn't start with a Standard MIDI header (MThd).".into());
    }
    let header_len = u32::from_be_bytes(chunk_header[4..8].try_into().unwrap()) as u64;
    if header_len < 6 { return Err("Malformed MIDI: the header chunk is shorter than six bytes.".into()); }
    let mut header = [0u8; 6];
    reader.read_exact(&mut header).map_err(|_| "Malformed MIDI: truncated header chunk.".to_string())?;
    let format = u16::from_be_bytes([header[0], header[1]]);
    let declared_tracks = u16::from_be_bytes([header[2], header[3]]);
    let division = u16::from_be_bytes([header[4], header[5]]);
    if format > 1 {
        return Err("MIDI format 2 (independent sequences) isn't supported; save as format 0 or 1.".into());
    }
    if declared_tracks == 0 { return Err("Malformed MIDI: the header declares zero tracks.".into()); }
    if division & 0x8000 != 0 {
        return Err("SMPTE-timed MIDI files aren't supported yet. Export with PPQ timing.".into());
    }
    let ppq = (division & 0x7fff) as f64;
    if ppq <= 0.0 { return Err("This MIDI has an invalid ticks-per-quarter value.".into()); }
    stream_skip(&mut reader, header_len - 6)?;

    let mut notes: Vec<Note> = Vec::new();
    // A small initial reservation reduces growth churn without reserving memory
    // proportional to a potentially huge MIDI file's byte size.
    let initial_note_capacity = (file_size / 4096).min(262_144).max(1) as usize;
    notes.try_reserve(initial_note_capacity).map_err(|_| {
        "Not enough memory to start parsing this MIDI. Close other applications or use a machine with more RAM.".to_string()
    })?;
    let mut tempo_events: Vec<(u64, u32)> = Vec::with_capacity(16);
    let mut parsed_tracks = 0u32;
    while parsed_tracks < u32::from(declared_tracks) {
        reader.read_exact(&mut chunk_header).map_err(|_| {
            format!("Malformed MIDI: found only {parsed_tracks} of {declared_tracks} declared track chunks.")
        })?;
        let chunk_len = u32::from_be_bytes(chunk_header[4..8].try_into().unwrap()) as u64;
        if &chunk_header[..4] == b"MTrk" {
            parse_midi_track(&mut reader, chunk_len, parsed_tracks, &mut notes, &mut tempo_events)?;
            parsed_tracks += 1;
        } else {
            // Permit non-track chunks between tracks while avoiding loading their data.
            stream_skip(&mut reader, chunk_len)?;
        }
    }

    if notes.is_empty() { return Err("This MIDI has no note events to chart.".into()); }
    tempo_events.sort_by_key(|x| x.0);
    tempo_events.dedup_by_key(|x| x.0);
    if tempo_events.first().map(|x| x.0) != Some(0) { tempo_events.insert(0, (0, 500_000)); }
    let tempo_segments = build_tempo_segments(&tempo_events, ppq);
    let mut max_ms: f64 = 0.0;
    for note in &mut notes {
        let start_tick = note.ms as u64;
        let end_tick = note.end_ms as u64;
        note.ms = ticks_ms_fast(start_tick, ppq, &tempo_segments);
        note.end_ms = ticks_ms_fast(end_tick, ppq, &tempo_segments);
        max_ms = max_ms.max(note.end_ms);
    }
    let bpm = 60_000_000.0 / tempo_events[0].1 as f64;
    let note_count = notes.len();
    Ok(MidiInfo { notes, note_count, bpm, tracks: parsed_tracks as usize, duration: max_ms })
}

fn build_tempo_segments(events: &[(u64, u32)], ppq: f64) -> Vec<(u64, u32, f64)> {
    let mut segments = Vec::with_capacity(events.len());
    let mut last_tick = 0u64;
    let mut elapsed_ms = 0.0;
    let mut us_per_beat = 500_000u32;
    for &(tick, next_tempo) in events {
        if tick > last_tick {
            elapsed_ms += (tick - last_tick) as f64 * us_per_beat as f64 / ppq / 1000.0;
            last_tick = tick;
        }
        us_per_beat = next_tempo;
        if segments.last().map(|s: &(u64, u32, f64)| s.0 == tick).unwrap_or(false) {
            if let Some(last) = segments.last_mut() { *last = (tick, us_per_beat, elapsed_ms); }
        } else {
            segments.push((tick, us_per_beat, elapsed_ms));
        }
    }
    if segments.is_empty() { segments.push((0, 500_000, 0.0)); }
    segments
}

fn ticks_ms_fast(tick: u64, ppq: f64, segments: &[(u64, u32, f64)]) -> f64 {
    let index = segments.partition_point(|segment| segment.0 <= tick).saturating_sub(1);
    let (start_tick, us_per_beat, start_ms) = segments[index];
    start_ms + tick.saturating_sub(start_tick) as f64 * us_per_beat as f64 / ppq / 1000.0
}

fn lane(p:u8, min:u8, max:u8, mapping:Mapping)->i32 {
    match mapping {
        Mapping::AccurateRanges => match p { 103=>3,104=>2,106=>1,107=>0,0..=63=>0,64..=68=>1,69..=74=>2,_=>3 },
        Mapping::Adaptive => {
            if max.saturating_sub(min)<=6 { (p%4) as i32 } else { (((p-min) as f64/(max-min) as f64*3.999).floor() as i32).clamp(0,3) }
        }
        Mapping::SmartBalance => (p as i32 - min as i32).rem_euclid(4),
    }
}
fn balanced_pitch_map(notes: &[Note]) -> [i32; 128] {
    let mut frequencies = [0u64; 128];
    for note in notes {
        frequencies[note.pitch as usize] = frequencies[note.pitch as usize].saturating_add(1);
    }
    let mut ranked: Vec<(u8, u64)> = frequencies.iter().enumerate()
        .filter_map(|(pitch, &count)| if count > 0 { Some((pitch as u8, count)) } else { None })
        .collect();
    // This matches MIDI2Psych's greedy histogram-balancing mapper: map common pitches
    // first, always breaking ties by lower pitch and then lower lane index.
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut lane_loads = [0u64; 4];
    let mut pitch_map = [0i32; 128];
    for (pitch, count) in ranked {
        let target = (0..4).min_by_key(|&lane_idx| lane_loads[lane_idx]).unwrap_or(0);
        pitch_map[pitch as usize] = target as i32;
        lane_loads[target] = lane_loads[target].saturating_add(count);
    }
    pitch_map
}

fn midi2psych_lane(note: &Note, pitch_map: &[i32; 128], dual: bool) -> i32 {
    pitch_map[note.pitch as usize] + if dual && note.player_two { 4 } else { 0 }
}

fn apply_midi2psych_placement(notes: &mut [Note], bpm: f64, dual: bool, pitch_map: &[i32; 128]) {
    if notes.len() < 2 { return; }
    // Sort in-place by lane and time, so collision placement is linear after the sort
    // and doesn't need a second 50-million-entry index array.
    if notes.len() > 10_000 {
        notes.par_sort_unstable_by(|a, b| {
            midi2psych_lane(a, pitch_map, dual).cmp(&midi2psych_lane(b, pitch_map, dual))
                .then(a.ms.total_cmp(&b.ms))
                .then(a.pitch.cmp(&b.pitch))
        });
    } else {
        notes.sort_unstable_by(|a, b| {
            midi2psych_lane(a, pitch_map, dual).cmp(&midi2psych_lane(b, pitch_map, dual))
                .then(a.ms.total_cmp(&b.ms))
                .then(a.pitch.cmp(&b.pitch))
        });
    }
    let section_ms = (60_000.0 / bpm.max(1.0)) * 4.0;
    let epsilon = 0.001;
    let mut start = 0usize;
    while start < notes.len() {
        let lane_id = midi2psych_lane(&notes[start], pitch_map, dual);
        let current_time = notes[start].ms;
        let mut end = start + 1;
        while end < notes.len()
            && midi2psych_lane(&notes[end], pitch_map, dual) == lane_id
            && (notes[end].ms - current_time).abs() < epsilon
        {
            end += 1;
        }
        if end - start > 1 && end < notes.len()
            && midi2psych_lane(&notes[end], pitch_map, dual) == lane_id
        {
            let next_time = notes[end].ms;
            let same_section = (current_time / section_ms).floor() == (next_time / section_ms).floor();
            if same_section && next_time > current_time {
                let step = (next_time - current_time) / ((end - start + 1) as f64);
                for idx in start..end {
                    let shifted_time = current_time + step * ((idx - start + 1) as f64);
                    let delta = shifted_time - notes[idx].ms;
                    notes[idx].ms = shifted_time;
                    // Keep the sustain duration unchanged while shifting note-on time.
                    notes[idx].end_ms += delta;
                }
            }
        }
        start = end;
    }
    if notes.len() > 10_000 {
        notes.par_sort_unstable_by(|a, b| a.ms.total_cmp(&b.ms).then(a.pitch.cmp(&b.pitch)));
    } else {
        notes.sort_unstable_by(|a, b| a.ms.total_cmp(&b.ms).then(a.pitch.cmp(&b.pitch)));
    }
}

fn mapped_base_lane(note: &Note, plan: &PendingExport, pitch_map: &[i32; 128], min: u8, max: u8) -> i32 {
    let mut mapped = if plan.style == Style::Midi2PsychMapping || plan.mapping == Mapping::SmartBalance {
        pitch_map[note.pitch as usize]
    } else {
        lane(note.pitch, min, max, plan.mapping)
    };
    if plan.style == Style::Stream {
        mapped = (mapped + ((note.ms / 150.0).floor() as i32 % 4)).rem_euclid(4);
    } else if plan.style == Style::ChordFriendly {
        mapped = lane(note.pitch, min, max, Mapping::Adaptive);
    }
    mapped
}

fn output_lane(note: &Note, plan: &PendingExport, pitch_map: &[i32; 128], min: u8, max: u8, must_hit_section: bool) -> i32 {
    let mapped = mapped_base_lane(note, plan, pitch_map, min, max);
    if plan.style == Style::Midi2PsychMapping && plan.dual {
        // Match MIDI2Psych's section-aware player/opponent orientation. The
        // section with more P1 notes is player-led; ties favour P1.
        if must_hit_section {
            if note.player_two { mapped + 4 } else { mapped }
        } else if note.player_two { mapped } else { mapped + 4 }
    } else {
        // Keep the legacy lane behavior for the other chart styles.
        let is_opponent = if plan.style == Style::Midi2PsychMapping {
            plan.dual && note.player_two
        } else {
            plan.dual && (note.channel == 1 || note.track % 2 == 1)
        };
        if is_opponent { mapped + 4 } else { mapped }
    }
}

fn section_for(ms: f64, section_ms: f64) -> usize {
    (ms / section_ms).floor().max(0.0) as usize
}

fn pitch_bounds(notes: &[Note]) -> (u8, u8) {
    let mut min_pitch = u8::MAX;
    let mut max_pitch = 0u8;
    for note in notes {
        min_pitch = min_pitch.min(note.pitch);
        max_pitch = max_pitch.max(note.pitch);
    }
    if notes.is_empty() { (0, 127) } else { (min_pitch, max_pitch) }
}

fn write_chart_json<W: Write>(
    writer: &mut W,
    plan: &PendingExport,
    notes: &[Note],
    pitch_map: &[i32; 128],
    min_pitch: u8,
    max_pitch: u8,
    compact_sections: bool,
) -> io::Result<()> {
    if notes.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No notes were selected."));
    }
    let section_ms = (60_000.0 / plan.bpm.max(1.0)) * 4.0;
    let first_note_section = section_for(notes[0].ms, section_ms);
    let last_note_section = section_for(notes[notes.len() - 1].ms, section_ms);
    // Split files start at their first used section rather than writing thousands of
    // leading empty sections. Unsplit charts retain the classic section-zero origin.
    let start_section = if compact_sections { first_note_section } else { 0 };

    writer.write_all(b"{\"song\":{\"song\":")?;
    serde_json::to_writer(&mut *writer, &plan.song)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(b",\"notes\":[")?;

    let mut note_idx = 0usize;
    let mut last_must_hit = true;
    for section_idx in start_section..=last_note_section {
        if section_idx > start_section { writer.write_all(b",")?; }
        let section_start = note_idx;
        while note_idx < notes.len() && section_for(notes[note_idx].ms, section_ms) == section_idx {
            note_idx += 1;
        }
        let section_end = note_idx;
        let must_hit_section = if plan.style == Style::Midi2PsychMapping && plan.dual {
            if section_start == section_end {
                last_must_hit
            } else {
                let p1_count = notes[section_start..section_end].iter().filter(|n| !n.player_two).count();
                let p2_count = section_end - section_start - p1_count;
                last_must_hit = p1_count >= p2_count;
                last_must_hit
            }
        } else {
            !plan.dual
        };
        writer.write_all(b"{\"sectionNotes\":[")?;
        let mut first_in_section = true;
        for note in &notes[section_start..section_end] {
            if !first_in_section { writer.write_all(b",")?; }
            first_in_section = false;
            let lane_id = output_lane(note, plan, pitch_map, min_pitch, max_pitch, must_hit_section);
            let duration = if plan.sustain { (note.end_ms - note.ms).max(0.0) } else { 0.0 };
            // Compact triples and a buffered writer avoid allocating a String per note
            // or constructing a multi-gigabyte JSON String before it can be saved.
            write!(writer, "[{},{},{}]", note.ms, lane_id, duration)?;
        }
        write!(
            writer,
            "],\"lengthInSteps\":16,\"mustHitSection\":{},\"gfSection\":false,\"altAnim\":false,\"typeOfSection\":0,\"changeBPM\":false,\"bpm\":{},\"sectionBeats\":4,\"crossFade\":false}}",
            must_hit_section, plan.bpm
        )?;
    }

    write!(writer, "],\"bpm\":{},\"needsVoices\":{},\"speed\":{},\"player1\":", plan.bpm, plan.needs_voices, plan.speed)?;
    serde_json::to_writer(&mut *writer, &plan.player).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(b",\"player2\":")?;
    serde_json::to_writer(&mut *writer, &plan.opponent).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(b",\"gfVersion\":\"gf\",\"stage\":")?;
    serde_json::to_writer(&mut *writer, &plan.stage).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(b",\"validScore\":true,\"events\":[],\"splashSkin\":\"noteSplashes\",\"arrowSkin\":\"\",\"specialAudioName\":\"\"}}")
}

struct LimitWriter<W> {
    inner: W,
    limit: u64,
    written: u64,
    exceeded: bool,
}
impl<W: Write> Write for LimitWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.is_empty() { return Ok(0); }
        let remaining = self.limit.saturating_sub(self.written);
        if remaining == 0 {
            self.exceeded = true;
            return Err(io::Error::new(io::ErrorKind::Other, "JSON_LIMIT_REACHED"));
        }
        let allowed = buffer.len().min(remaining.min(usize::MAX as u64) as usize);
        let count = self.inner.write(&buffer[..allowed])?;
        self.written = self.written.saturating_add(count as u64);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
fn unique_temp_path(parent: &Path, purpose: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(".midi-to-psych-{purpose}-{}-{nanos}-{sequence}.tmp", std::process::id()))
}

fn write_chart_file(
    path: &Path,
    plan: &PendingExport,
    notes: &[Note],
    pitch_map: &[i32; 128],
    min_pitch: u8,
    max_pitch: u8,
    compact_sections: bool,
    byte_limit: u64,
) -> Result<u64, String> {
    use std::fs::OpenOptions;
    let file = OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|e| format!("Couldn't create temporary JSON file: {e}"))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, LimitWriter {
        inner: file, limit: byte_limit, written: 0, exceeded: false,
    });
    let write_result = write_chart_json(&mut writer, plan, notes, pitch_map, min_pitch, max_pitch, compact_sections);
    let flush_result = writer.flush();
    let limit_hit = writer.get_ref().exceeded;
    drop(writer);
    if limit_hit {
        let _ = fs::remove_file(path);
        return Err(format!("SPLIT_REQUIRED:{}", byte_limit.saturating_add(1)));
    }
    if let Err(e) = write_result {
        let _ = fs::remove_file(path);
        return Err(format!("Could not serialize chart JSON: {e}"));
    }
    if let Err(e) = flush_result {
        let _ = fs::remove_file(path);
        return Err(format!("Could not finish chart JSON: {e}"));
    }
    fs::metadata(path).map(|m| m.len()).map_err(|e| {
        let _ = fs::remove_file(path);
        format!("Could not inspect chart JSON size: {e}")
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JsonCap { Mb250, Mb500, Gb1, Gb15, Gb2, Gb21 }
impl JsonCap {
    const ALL: [JsonCap; 6] = [JsonCap::Mb250, JsonCap::Mb500, JsonCap::Gb1, JsonCap::Gb15, JsonCap::Gb2, JsonCap::Gb21];
    fn label(self) -> &'static str {
        match self { Self::Mb250 => "250 MB", Self::Mb500 => "500 MB", Self::Gb1 => "1 GB", Self::Gb15 => "1.5 GB", Self::Gb2 => "2 GB", Self::Gb21 => "2.1 GB" }
    }
    // Decimal byte units are used so the labels match the limits shown in the UI.
    fn bytes(self) -> u64 {
        match self { Self::Mb250 => 250_000_000, Self::Mb500 => 500_000_000, Self::Gb1 => 1_000_000_000, Self::Gb15 => 1_500_000_000, Self::Gb2 => 2_000_000_000, Self::Gb21 => 2_100_000_000 }
    }
}
const MAX_JSON_BYTES: u64 = 2_100_000_000;

struct PendingExport {
    destination: PathBuf,
    notes: Vec<Note>,
    bpm: f64,
    speed: f32,
    song: String,
    player: String,
    opponent: String,
    stage: String,
    mapping: Mapping,
    style: Style,
    pitch_map: [i32; 128],
    dual: bool,
    sustain: bool,
    needs_voices: bool,
    inst_audio: PathBuf,
    vocals_audio: Option<PathBuf>,
    mod_name: String,
    include_template: bool,
    cap: JsonCap,
    estimated_bytes: u64,
}
fn estimate_json_bytes(notes: &[Note], bpm: f64) -> u64 {
    // The compact streaming writer emits roughly 20–30 bytes per note depending on
    // timestamp/duration widths. Keep the estimate lightweight without allocating
    // per-note strings or an in-memory section map.
    let section_ms = (60_000.0 / bpm.max(1.0)) * 4.0;
    let section_count = notes.iter()
        .map(|n| section_for(n.ms, section_ms) as u64)
        .max().unwrap_or(0).saturating_add(1);
    (notes.len() as u64).saturating_mul(28)
        .saturating_add(section_count.saturating_mul(192))
        .saturating_add(4096)
}

fn pretty_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.2} GB", bytes as f64 / 1_000_000_000.0) }
    else if bytes >= 1_000_000 { format!("{:.1} MB", bytes as f64 / 1_000_000.0) }
    else if bytes >= 1_000 { format!("{:.1} KB", bytes as f64 / 1_000.0) }
    else { format!("{bytes} bytes") }
}

type MidiLoadMessage = (bool, PathBuf, Result<MidiInfo, String>);

struct App {
    mode: Mode, mapping: Mapping, style: Style, p1: Option<PathBuf>, p2: Option<PathBuf>,
    song: String, output: Option<PathBuf>, player: String, opponent: String, stage: String,
    speed_input: String, sustain: bool, status: String, error: bool, info1: Option<MidiInfo>, info2: Option<MidiInfo>,
    inst_audio: Option<PathBuf>, vocals_audio: Option<PathBuf>, mod_name: String, include_template: bool,
    json_cap: JsonCap, pending_export: Option<PendingExport>, split_prompt: bool,
    load_rx: Option<Receiver<MidiLoadMessage>>,
}
impl Default for App {
    fn default()->Self { Self{mode:Mode::Single,mapping:Mapping::AccurateRanges,style:Style::Classic,p1:None,p2:None,
        song:"My Song".into(),output:None,player:"bf".into(),opponent:"dad".into(),stage:"stage".into(),speed_input:"2.5".into(),
        sustain:true,status:"Ready when you are. Load a MIDI and your instrumental audio.".into(),error:false,
        info1:None,info2:None,inst_audio:None,vocals_audio:None,mod_name:"My-MIDI-Mod".into(),include_template:false,
        json_cap:JsonCap::Gb21,pending_export:None,split_prompt:false,load_rx:None} }
}
impl App {
    fn start_midi_load(&mut self, first: bool, path: PathBuf) -> bool {
        if self.load_rx.is_some() { return false; }
        let display_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let thread_path = path.clone();
        let (tx, rx) = mpsc::channel::<MidiLoadMessage>();
        match std::thread::Builder::new().name("midi-file-loader".into()).spawn(move || {
            let result = parse_midi(&thread_path);
            let _ = tx.send((first, thread_path, result));
        }) {
            Ok(_) => {
                self.load_rx = Some(rx);
                self.status = format!("Loading and analyzing {display_name}. Large MIDI files are read progressively, so this may take a while.");
                self.error = false;
                true
            }
            Err(e) => {
                self.status = format!("Couldn't start the MIDI loader thread: {e}");
                self.error = true;
                false
            }
        }
    }

    fn choose(&mut self, first:bool) {
        if self.load_rx.is_some() { return; }
        #[cfg(target_os = "android")]
        {
            let _ = first;
            self.status = "Android file selection is not connected yet. This workflow only checks that the Rust code cross-compiles for Android; use the desktop build to import MIDI files.".into();
            self.error = true;
        }
        #[cfg(not(target_os = "android"))]
        {
            if let Some(path)=rfd::FileDialog::new().add_filter("MIDI files",&["mid","midi"]).pick_file() {
                self.start_midi_load(first, path);
            }
        }
    }

    fn poll_midi_load(&mut self, ctx: &egui::Context) {
        let received = match self.load_rx.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(message) => Some(Ok(message)),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err(())),
            },
            None => return,
        };
        match received {
            None => ctx.request_repaint_after(Duration::from_millis(150)),
            Some(Err(())) => {
                self.load_rx = None;
                self.status = "The MIDI loader stopped unexpectedly before returning a result.".into();
                self.error = true;
            }
            Some(Ok((first, path, result))) => {
                self.load_rx = None;
                match result {
                    Ok(info) => {
                        if first { self.p1 = Some(path); self.info1 = Some(info); }
                        else { self.p2 = Some(path); self.info2 = Some(info); }
                        self.status = "MIDI loaded and analyzed successfully.".into();
                        self.error = false;
                    }
                    Err(message) => {
                        self.status = message;
                        self.error = true;
                    }
                }
            }
        }
    }
    fn choose_audio(&mut self, instrumental: bool) {
        #[cfg(target_os = "android")]
        {
            let _ = instrumental;
            self.status = "Android file selection is not connected yet. The Android workflow is a cross-compile check, not a usable APK.".into();
            self.error = true;
        }
        #[cfg(not(target_os = "android"))]
        {
            let title = if instrumental { "Select instrumental audio (Inst.ogg)" } else { "Select vocals audio (optional)" };
            if let Some(path) = rfd::FileDialog::new().set_title(title).add_filter("OGG audio", &["ogg"]).pick_file() {
                if !path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("ogg")).unwrap_or(false) {
                    self.status = "Please choose an .ogg audio file.".into();
                    self.error = true;
                    return;
                }
                if instrumental { self.inst_audio = Some(path); }
                else { self.vocals_audio = Some(path); }
                self.status = if instrumental { "Instrumental audio selected." } else { "Vocals audio selected; it will be exported as Voices.ogg for Psych Engine." }.into();
                self.error = false;
            }
        }
    }
    fn export(&mut self) {
        if self.load_rx.is_some() {
            self.status = "Wait for the selected MIDI to finish loading before exporting.".into();
            self.error = true;
            return;
        }
        if self.p1.is_none() {
            self.status="Select a MIDI file before exporting the chart.".into();self.error=true;return;
        }
        if self.mode==Mode::Dual && self.p2.is_none() {
            self.status="Two-MIDI mode needs both MIDI slots. Add the second MIDI first.".into();self.error=true;return;
        }
        if self.inst_audio.is_none() {
            self.status="Instrumental audio is required. Select your Inst.ogg file first. Vocals are optional.".into();self.error=true;return;
        }
        if self.song.trim().is_empty() {
            self.status="Enter a song name before exporting.".into();self.error=true;return;
        }
        let speed = match self.speed_input.trim().parse::<f32>() {
            Ok(v) if v.is_finite() && (1.0..=1000.0).contains(&v) => v,
            _ => { self.status="Scroll speed must be a number from 1 to 1000.".into();self.error=true;return; }
        };
        // Export consumes the large in-memory note vector to avoid duplicating it.
        // If the user exports again, reload off-thread rather than freezing the UI.
        if self.info1.as_ref().map(|info| info.notes.is_empty()).unwrap_or(true) {
            let path = match self.p1.as_ref() { Some(path) => path.clone(), None => return };
            if self.start_midi_load(true, path) {
                self.status = "Reloading MIDI 1 in the background after the previous export. Once it finishes, press EXPORT again.".into();
                self.error = false;
            }
            return;
        }
        if self.mode == Mode::Dual && self.info2.as_ref().map(|info| info.notes.is_empty()).unwrap_or(true) {
            let path = match self.p2.as_ref() { Some(path) => path.clone(), None => return };
            if self.start_midi_load(false, path) {
                self.status = "Reloading MIDI 2 in the background after the previous export. Once it finishes, press EXPORT again.".into();
                self.error = false;
            }
            return;
        }
        #[cfg(target_os = "android")]
        let destination: PathBuf = {
            self.status = "Android folder selection/export is not connected yet. This Android job only verifies cross-compilation; it does not create an installable APK.".into();
            self.error = true;
            return;
        };
        #[cfg(not(target_os = "android"))]
        let destination = match rfd::FileDialog::new()
            .set_directory(self.output.as_deref().unwrap_or(Path::new(".")))
            .set_title("Choose where to create the Psych Engine mod folder")
            .pick_folder() {
                Some(p) => p,
                None => return,
            };
        // Notes are already loaded (or the async reload above returned before reaching here).
        let mut bpm = 120.0;
        let mut notes = if let Some(info) = self.info1.as_mut() {
            bpm = info.bpm;
            std::mem::take(&mut info.notes)
        } else { Vec::new() };
        if self.mode == Mode::Dual {
            if let Some(info) = self.info2.as_mut() {
                let mut second_notes = std::mem::take(&mut info.notes);
                for note in &mut second_notes { note.player_two = true; }
                notes.append(&mut second_notes);
            }
        }
        let uses_histogram_mapper = self.style == Style::Midi2PsychMapping || self.mapping == Mapping::SmartBalance;
        let pitch_map = if uses_histogram_mapper { balanced_pitch_map(&notes) } else { [0i32; 128] };
        if self.style == Style::Midi2PsychMapping {
            apply_midi2psych_placement(&mut notes, bpm, self.mode == Mode::Dual, &pitch_map);
        } else if notes.len() > 10_000 {
            notes.par_sort_unstable_by(|a, b| a.ms.total_cmp(&b.ms).then(a.pitch.cmp(&b.pitch)));
        } else {
            notes.sort_unstable_by(|a,b| a.ms.total_cmp(&b.ms).then(a.pitch.cmp(&b.pitch)));
        }
        let inst_audio=match self.inst_audio.clone(){Some(p)=>p,None=>return};
        let mut plan=PendingExport {
            destination, notes, bpm, speed, song:self.song.trim().to_string(), player:self.player.trim().to_string(),
            opponent:self.opponent.trim().to_string(), stage:self.stage.trim().to_string(), mapping:self.mapping,
            style:self.style, pitch_map, dual:self.mode==Mode::Dual, sustain:self.sustain, needs_voices:self.vocals_audio.is_some(), inst_audio,
            vocals_audio:self.vocals_audio.clone(), mod_name:self.mod_name.trim().to_string(),
            include_template:self.include_template, cap:self.json_cap, estimated_bytes:0,
        };
        plan.estimated_bytes=estimate_json_bytes(&plan.notes,plan.bpm);
        if plan.estimated_bytes >= plan.cap.bytes() || plan.estimated_bytes >= MAX_JSON_BYTES {
            self.open_split_prompt(plan);
            return;
        }
        match self.make_mod(&plan,false,false) {
            Ok((folder,parts,total_bytes))=>self.finish_export_status(&plan,folder,parts,total_bytes),
            Err(e) if e.starts_with("SPLIT_REQUIRED:") => {
                let actual=e.trim_start_matches("SPLIT_REQUIRED:").parse::<u64>().unwrap_or(plan.estimated_bytes);
                plan.estimated_bytes=actual;
                self.open_split_prompt(plan);
            }
            Err(e)=>{self.status=format!("Export failed: {e}");self.error=true;}
        }
    }

    fn open_split_prompt(&mut self, plan: PendingExport) {
        self.status=format!("Chart JSON is estimated at {}. Choose whether to split it.",pretty_bytes(plan.estimated_bytes));
        self.error=false;
        self.pending_export=Some(plan);
        self.split_prompt=true;
    }

    fn finish_export_status(&mut self, plan:&PendingExport, folder:PathBuf, parts:usize, total_bytes:u64) {
        self.output=Some(plan.destination.clone());
        let id=song_id(&plan.song);
        let split_note=if parts>1 {
            format!("\nJSON split into {parts} files ({} total).\nImportant: stock Psych Engine does not automatically merge split chart files; see data/{id}/{id}.split-manifest.json.",pretty_bytes(total_bytes))
        } else { String::new() };
        self.status=format!(
            "Psych Engine mod exported successfully!\n{}\n\nChart: data/{id}/{id}.json\nAudio: songs/{id}/Inst.ogg{}\nWeek: weeks/week-{id}.json{}",
            folder.display(), if plan.needs_voices { " + Voices.ogg" } else { " (instrumental only)" }, split_note
        );
        self.error=false;
    }

    fn make_mod(&self, plan: &PendingExport, split: bool, ignore_selected_cap: bool) -> Result<(PathBuf, usize, u64), String> {
        let folder = plan.destination.join(safe_name(&plan.mod_name));
        if folder.exists() {
            return Err(format!("Mod folder already exists: {}. Choose another mod name or destination so existing files aren't overwritten.", folder.display()));
        }
        if plan.notes.is_empty() { return Err("No notes were selected.".into()); }

        let pitch_map = &plan.pitch_map;
        let (min_pitch, max_pitch) = pitch_bounds(&plan.notes);
        let single_temp: Option<(PathBuf, u64)> = if !split {
            // Stream to a temporary disk file and stop as soon as a cap is exceeded.
            // The complete JSON is never materialized as a single giant String.
            let temp_path = unique_temp_path(&plan.destination, "chart");
            let byte_limit = if ignore_selected_cap { MAX_JSON_BYTES } else { plan.cap.bytes().min(MAX_JSON_BYTES) };
            match write_chart_file(
                &temp_path, plan, &plan.notes, pitch_map, min_pitch, max_pitch, false, byte_limit,
            ) {
                Ok(bytes) => Some((temp_path, bytes)),
                Err(e) if e.starts_with("SPLIT_REQUIRED:") => return Err(e),
                Err(e) => return Err(e),
            }
        } else { None };

        let result = (|| -> Result<(usize, u64), String> {
            let template = Path::new("assets/Andres-First-Mod");
            if plan.include_template && template.exists() {
                copy_dir(template, &folder).map_err(|e| e.to_string())?;
            } else {
                fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
            }
            let id = song_id(&plan.song);
            let data_dir = folder.join("data").join(&id);
            fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;

            let mut part_metadata: Vec<serde_json::Value> = Vec::new();
            let mut total_bytes = 0u64;
            let mut part_count = 0usize;
            if let Some((temp_path, bytes)) = single_temp.as_ref() {
                let final_path = data_dir.join(format!("{id}.json"));
                if final_path.exists() { fs::remove_file(&final_path).map_err(|e| e.to_string())?; }
                fs::rename(temp_path, &final_path).map_err(|e| format!("Couldn't move completed chart JSON into the mod folder: {e}"))?;
                total_bytes = *bytes;
                part_count = 1;
                part_metadata.push(serde_json::json!({
                    "file": format!("{id}.json"), "bytes": bytes, "notes": plan.notes.len(),
                    "firstNoteMs": plan.notes.first().map(|n| n.ms),
                    "lastNoteMs": plan.notes.last().map(|n| n.ms)
                }));
            } else {
                let cap = plan.cap.bytes().min(MAX_JSON_BYTES);
                // Compact streaming output averages roughly 20–30 bytes per note.
                // Start close to the selected cap; retry a chunk at half size if an
                // unusually long timestamp/duration makes it exceed the byte limit.
                let initial = (cap / 28).max(1).min(10_000_000) as usize;
                let mut chunk_size = initial.max(1);
                let mut start = 0usize;
                while start < plan.notes.len() {
                    let mut end = start.saturating_add(chunk_size).min(plan.notes.len());
                    let (temp_path, bytes) = loop {
                        let candidate_path = unique_temp_path(&data_dir, "part");
                        match write_chart_file(
                            &candidate_path, plan, &plan.notes[start..end], pitch_map,
                            min_pitch, max_pitch, true, cap,
                        ) {
                            Ok(bytes) => break (candidate_path, bytes),
                            Err(e) if e.starts_with("SPLIT_REQUIRED:") => {
                                if end - start <= 1 {
                                    return Err(format!("Even a one-note JSON part exceeds the selected {} limit because of chart metadata.", plan.cap.label()));
                                }
                                end = start + (end - start) / 2;
                                chunk_size = (end - start).max(1);
                            }
                            Err(e) => return Err(e),
                        }
                    };

                    let file_name = if part_count == 0 {
                        format!("{id}.json")
                    } else {
                        format!("{id}-part-{:03}.json", part_count + 1)
                    };
                    let final_path = data_dir.join(&file_name);
                    if final_path.exists() { fs::remove_file(&final_path).map_err(|e| e.to_string())?; }
                    fs::rename(&temp_path, &final_path).map_err(|e| format!("Couldn't finalize JSON part: {e}"))?;

                    let slice = &plan.notes[start..end];
                    part_metadata.push(serde_json::json!({
                        "file": file_name, "bytes": bytes, "notes": slice.len(),
                        "firstNoteMs": slice.first().map(|n| n.ms),
                        "lastNoteMs": slice.last().map(|n| n.ms)
                    }));
                    total_bytes = total_bytes.saturating_add(bytes);
                    part_count += 1;
                    start = end;
                }
                if part_count > 1 {
                    let manifest = serde_json::json!({
                        "format": "midi-to-psych-rust-split-v1",
                        "song": plan.song,
                        "maxPartBytes": cap,
                        "partCount": part_count,
                        "parts": part_metadata,
                        "compatibilityWarning": "Stock Psych Engine loads only the primary song JSON and does not automatically merge these parts. A split-aware loader or merge step is required to play the complete chart as one song."
                    });
                    fs::write(
                        data_dir.join(format!("{id}.split-manifest.json")),
                        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?,
                    ).map_err(|e| e.to_string())?;
                }
            }

            let song_dir = folder.join("songs").join(&id);
            fs::create_dir_all(&song_dir).map_err(|e| e.to_string())?;
            fs::copy(&plan.inst_audio, song_dir.join("Inst.ogg"))
                .map_err(|e| format!("Couldn't copy instrumental audio: {e}"))?;
            if let Some(vocals) = &plan.vocals_audio {
                fs::copy(vocals, song_dir.join("Voices.ogg"))
                    .map_err(|e| format!("Couldn't copy vocals audio: {e}"))?;
            }

            let weeks_dir = folder.join("weeks");
            fs::create_dir_all(&weeks_dir).map_err(|e| e.to_string())?;
            let week = serde_json::json!({
                "songs": [[id, plan.opponent, [146, 113, 253]]],
                "hiddenUntilUnlocked": false,
                "hideFreeplay": false,
                "weekBackground": plan.stage,
                "difficulties": "",
                "weekCharacters": [plan.opponent, plan.player, "gf"],
                "storyName": format!("{} Week", plan.song),
                "weekName": format!("{} Week", plan.song),
                "hideStoryMode": false,
                "weekBefore": "tutorial",
                "startUnlocked": true
            });
            let week_path = weeks_dir.join(format!("week-{id}.json"));
            fs::write(&week_path, serde_json::to_string_pretty(&week).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

            let pack = serde_json::json!({
                "name": plan.mod_name.trim(),
                "description": format!("MIDI chart pack for {}", plan.song),
                "restart": false,
                "runsGlobally": false,
                "color": [170, 0, 255],
                "iconFramerate": 10
            });
            fs::write(folder.join("pack.json"), serde_json::to_string_pretty(&pack).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let split_readme = if part_count > 1 {
                format!("\n\nSPLIT CHART WARNING: The primary chart is {id}.json and additional files are named {id}-part-002.json, etc. Stock Psych Engine does not automatically merge them. See data/{id}/{id}.split-manifest.json for the file list and note ranges; a split-aware loader or merge step is required for the entire chart.")
            } else { String::new() };
            fs::write(folder.join("MIDI-TO-PSYCH-README.txt"), format!(
                "Generated by MIDI to Psych Rust\nMod: {mod_name}\nSong: {song}\nScroll speed: {speed}\nMapping: {mapping:?}\nStyle: {style}\nVocals: {vocals}\nJSON size cap: {cap}\n\nInstall by placing this mod folder in your Psych Engine mods directory. The chart is in data/{id}/{id}.json, audio is in songs/{id}/, and the week file is in weeks/week-{id}.json. Vocals, when supplied, are exported as Voices.ogg (Psych Engine's expected filename).{split_readme}",
                mod_name=plan.mod_name, song=plan.song, speed=plan.speed, mapping=plan.mapping,
                style=plan.style.label(), cap=plan.cap.label(),
                vocals=if plan.vocals_audio.is_some() { "included" } else { "not included (instrumental-only)" },
                id=id, split_readme=split_readme
            )).map_err(|e| e.to_string())?;
            Ok((part_count, total_bytes))
        })();

        match result {
            Ok((parts, total_bytes)) => Ok((folder, parts, total_bytes)),
            Err(e) => {
                let _ = fs::remove_dir_all(&folder);
                if let Some((temp_path, _)) = single_temp.as_ref() { let _ = fs::remove_file(temp_path); }
                Err(e)
            }
        }
    }

    fn resolve_split_prompt(&mut self, split_now:bool) {
        self.split_prompt=false;
        let plan=match self.pending_export.take(){Some(p)=>p,None=>return};
        match self.make_mod(&plan,split_now,!split_now) {
            Ok((folder,parts,total_bytes))=>self.finish_export_status(&plan,folder,parts,total_bytes),
            Err(e) if e.starts_with("SPLIT_REQUIRED:") => {
                let bytes=e.trim_start_matches("SPLIT_REQUIRED:").parse::<u64>().unwrap_or(plan.estimated_bytes);
                self.status=format!("The serialized chart reached {} and exceeds the 2.1 GB maximum. Choose Yes Split on the next export attempt.",pretty_bytes(bytes));
                self.error=true;
            }
            Err(e)=>{self.status=format!("Export failed: {e}");self.error=true;}
        }
    }

}
fn song_id(s:&str)->String { safe_name(s).to_lowercase() }
fn safe_name(s:&str)->String { let x:String=s.chars().map(|c|if c.is_ascii_alphanumeric()||c=='-'||c=='_' {c}else if c.is_whitespace(){'-'}else{'_'}).collect(); let x=x.trim_matches('-').to_string(); if x.is_empty(){"Untitled".into()}else{x} }
fn copy_dir(src:&Path,dst:&Path)->std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {let e=entry?;let target=dst.join(e.file_name());if e.file_type()?.is_dir(){copy_dir(&e.path(),&target)?;}else{fs::copy(e.path(),target)?;}}
    Ok(())
}
impl eframe::App for App {
    fn update(&mut self, ctx:&egui::Context, _frame:&mut eframe::Frame) {
        self.poll_midi_load(ctx);
        let visuals=egui::Visuals::dark();ctx.set_visuals(visuals);
        egui::CentralPanel::default().show(ctx,|ui|{
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui|{
                ui.vertical(|ui|{
                    ui.heading(egui::RichText::new("MIDI → PSYCH").size(30.0).strong().color(egui::Color32::from_rgb(139,113,255)));
                    ui.label("RUST EDITION  •  MIDI MAPPER + MOD BUILDER");
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center),|ui|{ui.label("v0.1.0");});
            });
            ui.add_space(10.0);
            ui.group(|ui|{
                ui.heading("01  •  Choose MIDI mode");
                ui.horizontal(|ui|{
                    ui.selectable_value(&mut self.mode,Mode::Single,"Single MIDI Only");
                    ui.selectable_value(&mut self.mode,Mode::Dual,"2 MIDIs");
                });
                ui.label(if self.mode==Mode::Single {"Y = original MIDI-to-chart mode. Load one MIDI."}else{"N / two-MIDI path = combine two MIDI sources into one chart."});
                ui.add_space(6.0);
                let midi_loading = self.load_rx.is_some();
                ui.horizontal(|ui|{
                    if ui.add_enabled(!midi_loading, egui::Button::new("＋ Select MIDI 1")).clicked(){self.choose(true);}
                    if ui.add_enabled(!midi_loading, egui::Button::new("＋ Select MIDI 2")).clicked(){self.choose(false);}
                    if midi_loading { ui.spinner(); ui.label("Reading MIDI…"); }
                    else if self.mode==Mode::Single {ui.label("MIDI 2 is ignored in Single mode.");}
                });
                ui.label(format!("MIDI 1: {}",self.p1.as_ref().map(|p|p.file_name().unwrap_or_default().to_string_lossy().to_string()).unwrap_or("Not selected".into())));
                ui.label(format!("MIDI 2: {}",self.p2.as_ref().map(|p|p.file_name().unwrap_or_default().to_string_lossy().to_string()).unwrap_or("Not selected".into())));
                if let Some(i)=&self.info1 {ui.small(format!("MIDI 1 • {} tracks • {} notes • {:.1} BPM • {:.1}s",i.tracks,i.note_count,i.bpm,i.duration/1000.0));}
                if self.mode==Mode::Dual {if let Some(i)=&self.info2 {ui.small(format!("MIDI 2 • {} tracks • {} notes • {:.1} BPM • {:.1}s",i.tracks,i.note_count,i.bpm,i.duration/1000.0));}}
            });
            ui.add_space(8.0);
            ui.columns(2,|cols|{
                cols[0].group(|ui|{
                    ui.heading("02  •  Chart style");
                    ui.label("Mapper");
                    egui::ComboBox::from_id_salt("mapper").selected_text(match self.mapping{Mapping::AccurateRanges=>"MIDI-to-chart accurate ranges",Mapping::Adaptive=>"Adaptive pitch range",Mapping::SmartBalance=>"Smart pitch balance"}).show_ui(ui,|ui|{
                        ui.selectable_value(&mut self.mapping,Mapping::AccurateRanges,"MIDI-to-chart accurate ranges");
                        ui.selectable_value(&mut self.mapping,Mapping::Adaptive,"Adaptive pitch range");
                        ui.selectable_value(&mut self.mapping,Mapping::SmartBalance,"Smart pitch balance");
                    });
                    ui.label("Play style");
                    egui::ComboBox::from_id_salt("style").selected_text(self.style.label()).show_ui(ui,|ui|{
                        ui.selectable_value(&mut self.style,Style::Classic,"Classic");
                        ui.selectable_value(&mut self.style,Style::Stream,"Stream");
                        ui.selectable_value(&mut self.style,Style::ChordFriendly,"Chord-friendly");
                        ui.selectable_value(&mut self.style,Style::Midi2PsychMapping,"Midi2Psych Mapping");
                    });
                    if self.style == Style::Midi2PsychMapping {
                        ui.small("Uses MIDI2Psych histogram balancing and same-lane collision placement.");
                    }
                    ui.checkbox(&mut self.sustain,"Preserve sustain durations");
                    ui.horizontal(|ui| {
                        ui.label("Scroll speed (1–1000)");
                        ui.add(egui::TextEdit::singleline(&mut self.speed_input).desired_width(100.0).hint_text("e.g. 2.5"));
                    });
                });
                cols[1].group(|ui|{
                    ui.heading("03  •  Song metadata");
                    ui.label("Song name");ui.text_edit_singleline(&mut self.song);
                    ui.label("Player 1");ui.text_edit_singleline(&mut self.player);
                    ui.label("Player 2");ui.text_edit_singleline(&mut self.opponent);
                    ui.label("Stage");ui.text_edit_singleline(&mut self.stage);
                    ui.separator();
                    ui.label("Mod folder name");ui.text_edit_singleline(&mut self.mod_name);
                    ui.checkbox(&mut self.include_template,"Start from Andres-First-Mod template (if installed)");
                    ui.separator();
                    ui.label("JSON file size cap (prompts to split)");
                    egui::ComboBox::from_id_salt("json-size-cap").selected_text(self.json_cap.label()).show_ui(ui,|ui|{
                        for cap in JsonCap::ALL { ui.selectable_value(&mut self.json_cap,cap,cap.label()); }
                    });
                    ui.small("Caps: 250 MB, 500 MB, 1 GB, 1.5 GB, 2 GB, 2.1 GB. Absolute maximum: 2.1 GB.");
                    ui.small("The export includes data JSON, song audio, and a Week JSON file.");
                });
            });
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.heading("04  •  Song audio");
                ui.horizontal(|ui| {
                    if ui.button("＋ Select Inst.ogg (required)").clicked() { self.choose_audio(true); }
                    if ui.button("＋ Select Vocals.ogg (optional)").clicked() { self.choose_audio(false); }
                });
                ui.label(format!("Instrumental: {}", self.inst_audio.as_ref().map(|p|p.file_name().unwrap_or_default().to_string_lossy().to_string()).unwrap_or("Not selected — required".into())));
                ui.label(format!("Vocals: {}", self.vocals_audio.as_ref().map(|p|p.file_name().unwrap_or_default().to_string_lossy().to_string()).unwrap_or("Not selected — instrumental-only export".into())));
                ui.small("Vocals are written as Voices.ogg, the filename expected by Psych Engine.");
            });
            ui.add_space(10.0);
            ui.horizontal(|ui|{
                if ui.add_sized([240.0,42.0],egui::Button::new(egui::RichText::new("EXPORT MOD + WEEK JSON").strong())).clicked(){self.export();}
                if ui.add_enabled(self.load_rx.is_none(), egui::Button::new("Clear MIDI selections")).clicked(){self.p1=None;self.p2=None;self.info1=None;self.info2=None;self.status="MIDI selections cleared.".into();self.error=false;}
            });
            ui.add_space(8.0);
            ui.group(|ui|{
                ui.heading(if self.error {"Console • Error"}else{"Console • Status"});
                ui.label(egui::RichText::new(&self.status).color(if self.error{egui::Color32::from_rgb(255,110,130)}else{egui::Color32::from_rgb(130,220,170)}));
            });
            ui.add_space(4.0);
            ui.small("Note: MIDI pitch mapping is an approximation; ordinary MIDI files don't contain the original FNF arrow-lane metadata.");
            });
        });
        if self.split_prompt {
            let (cap_label,estimated,hard_max)=self.pending_export.as_ref().map(|p| {
                let label = if p.cap.bytes() < MAX_JSON_BYTES { p.cap.label() } else { "2.1 GB maximum" };
                (label, p.estimated_bytes, p.estimated_bytes >= MAX_JSON_BYTES)
            }).unwrap_or(("2.1 GB",0,false));
            let mut split_now=false;
            let mut not_yet=false;
            egui::Window::new("JSON size limit reached")
                .collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER,[0.0,0.0]).show(ctx,|ui|{
                    ui.set_min_width(390.0);
                    ui.label(egui::RichText::new(format!("Hey! You reached {cap_label}.")).strong());
                    ui.label("You can split JSONs now, or the file may become too large to read.");
                    ui.label("You have until 2.1 GB or it may not make it.");
                    ui.label(format!("Estimated chart JSON size: {}",pretty_bytes(estimated)));
                    ui.add_space(4.0);
                    if hard_max { ui.colored_label(egui::Color32::from_rgb(255,145,115),"No Later will try one file, but writing stops at the 2.1 GB maximum."); }
                    ui.add_space(8.0);
                    ui.horizontal(|ui|{
                        if ui.add_sized([130.0,34.0],egui::Button::new(egui::RichText::new("Yes Split").strong())).clicked(){split_now=true;}
                        if ui.add_sized([130.0,34.0],egui::Button::new("No Later")).clicked(){not_yet=true;}
                    });
                });
            if split_now { self.resolve_split_prompt(true); }
            else if not_yet { self.resolve_split_prompt(false); }
        }
    }
}
fn main()->eframe::Result<()> {
    let options=eframe::NativeOptions{viewport:egui::ViewportBuilder::default().with_inner_size([1000.0,900.0]).with_min_inner_size([760.0,650.0]),..Default::default()};
    eframe::run_native("MIDI to Psych Rust",options,Box::new(|_cc|Ok(Box::<App>::default())))
}
