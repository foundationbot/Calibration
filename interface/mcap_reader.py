#!/usr/bin/env python3
"""Minimal stdlib-only MCAP reader for OKVIS session recordings (#214 / #27).

Reads the files okvis2x_app_record --mcap writes. Deliberately NOT a general
MCAP implementation — it supports exactly what the house recorder produces and
what liveviz needs to consume, with zero dependencies (the same constraint as
the rest of liveviz: stdlib only, agent-inspectable):

  - uncompressed chunks (the recorder compiles compression out on purpose;
    a compressed chunk raises with a message saying which file/compression)
  - JSON channels with jsonschema schemas, plus schema-less png/jpeg channels
  - Metadata records (run_info)
  - TRUNCATION-TOLERANT linear scan: a recorder killed hard (kill -9, power)
    leaves no footer — this reader never needs one. It walks records from the
    front and stops cleanly at the first partial record, which is exactly the
    crash-safety contract of #213 (at most the unflushed tail is lost).

CLI:
  python3 mcap_reader.py session.mcap                    # channel summary
  python3 mcap_reader.py session.mcap --topic /imu0      # messages as JSON lines
  python3 mcap_reader.py session.mcap --topic /cam0/image --dump-dir out/  # payloads
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field

MAGIC = b"\x89MCAP0\r\n"

# Record opcodes (MCAP spec)
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_METADATA = 0x0C
OP_DATA_END = 0x0F


def _string(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    return buf[off: off + n].decode("utf-8", errors="replace"), off + n


@dataclass
class Channel:
    id: int
    topic: str
    message_encoding: str
    schema_name: str = ""
    schema_encoding: str = ""
    schema_data: bytes = b""
    message_count: int = 0


@dataclass
class McapSession:
    """Parsed session: channels by topic, metadata, and a message list or iterator."""
    channels: dict = field(default_factory=dict)  # topic -> Channel
    metadata: dict = field(default_factory=dict)  # name -> {k: v}
    truncated: bool = False  # file ended mid-record (recorder killed hard)


def _iter_records(buf: bytes, base: int):
    """Yield (opcode, body_view, body file offset) from a records blob that starts at file
    offset `base`; stop at a partial record."""
    off = 0
    end = len(buf)
    while off + 9 <= end:
        opcode = buf[off]
        (length,) = struct.unpack_from("<Q", buf, off + 1)
        body_start = off + 9
        if body_start + length > end:
            raise _Truncated(base + off)
        yield opcode, memoryview(buf)[body_start: body_start + length], base + body_start
        off = body_start + length
    if off != end:
        raise _Truncated(base + off)


class _Truncated(Exception):
    def __init__(self, at: int):
        super().__init__(f"partial record at byte {at}")
        self.at = at


def read_session(path: str, on_message=None, offsets: bool = False) -> McapSession:
    """Parse the file. If on_message is given it is called as
    on_message(channel: Channel, log_time_ns: int, publish_time_ns: int,
    data: memoryview) for every message, in file order. With offsets=True it also gets
    the payload's byte offset in the file as a fifth argument, so a caller can index
    large payloads and read them back later instead of holding them (chunks are
    uncompressed, so the payload's bytes sit in the file as-is)."""
    # mmap, not read(): a session with cameras runs ~3.7 GB/min, so real files
    # are tens of GB — far beyond RAM on the robot. The OS pages the mapping in
    # and out as the linear scan walks it; struct.unpack_from and memoryview
    # slicing work on mmap objects unchanged.
    import mmap as _mmap

    fh = open(path, "rb")
    try:
        buf = _mmap.mmap(fh.fileno(), 0, prot=_mmap.PROT_READ)
    except (ValueError, OSError):  # zero-length file cannot be mapped
        fh.close()
        raise ValueError(f"{path}: not an MCAP file (empty)")
    if buf[:len(MAGIC)] != MAGIC:
        buf.close()
        fh.close()
        raise ValueError(f"{path}: not an MCAP file (bad magic)")

    s = McapSession()
    by_id: dict = {}
    schemas: dict = {}

    def handle(opcode: int, body, at: int) -> None:
        if opcode == OP_SCHEMA:
            b = bytes(body)
            (sid,) = struct.unpack_from("<H", b, 0)
            name, off = _string(b, 2)
            enc, off = _string(b, off)
            (dn,) = struct.unpack_from("<I", b, off)
            schemas[sid] = (name, enc, b[off + 4: off + 4 + dn])
        elif opcode == OP_CHANNEL:
            b = bytes(body)
            cid, sid = struct.unpack_from("<HH", b, 0)
            topic, off = _string(b, 4)
            enc, off = _string(b, off)
            if cid not in by_id:
                ch = Channel(id=cid, topic=topic, message_encoding=enc)
                if sid in schemas:
                    ch.schema_name, ch.schema_encoding, ch.schema_data = schemas[sid]
                by_id[cid] = ch
                s.channels[topic] = ch
        elif opcode == OP_MESSAGE:
            cid, seq = struct.unpack_from("<HI", body, 0)
            log_t, pub_t = struct.unpack_from("<QQ", body, 6)
            ch = by_id.get(cid)
            if ch is None:
                return
            ch.message_count += 1
            if on_message is not None:
                if offsets:
                    on_message(ch, log_t, pub_t, body[22:], at + 22)
                else:
                    on_message(ch, log_t, pub_t, body[22:])
        elif opcode == OP_METADATA:
            b = bytes(body)
            name, off = _string(b, 0)
            (map_len,) = struct.unpack_from("<I", b, off)
            off += 4
            stop_at = off + map_len
            kv = {}
            while off < stop_at:
                k, off = _string(b, off)
                v, off = _string(b, off)
                kv[k] = v
            s.metadata[name] = kv
        elif opcode == OP_CHUNK:
            b = bytes(body)
            # start/end time (16) + uncompressed_size (8) + crc (4)
            (usize,) = struct.unpack_from("<Q", b, 16)
            compression, off = _string(b, 28)
            (blob_len,) = struct.unpack_from("<Q", b, off)
            off += 8
            blob = b[off: off + blob_len]
            if compression not in ("", "none"):
                raise ValueError(
                    f"{path}: chunk compressed with '{compression}' — the house "
                    "recorder writes uncompressed chunks; use `mcap` CLI for this file")
            del usize
            for op2, body2, at2 in _iter_records(blob, at + off):
                handle(op2, body2, at2)

    off = len(MAGIC)
    end = len(buf)
    try:
        while off + 9 <= end:
            opcode = buf[off]
            (length,) = struct.unpack_from("<Q", buf, off + 1)
            body_start = off + 9
            if body_start + length > end:
                raise _Truncated(off)
            if opcode in (OP_FOOTER, OP_DATA_END):
                # Footer/DataEnd start the summary tail; everything the reader
                # needs was in the data section. (Summary repeats channels.)
                off = body_start + length
                continue
            handle(opcode, memoryview(buf)[body_start: body_start + length], body_start)
            off = body_start + length
    except _Truncated:
        s.truncated = True
    finally:
        # close() on an mmap with live memoryviews raises BufferError, and in a finally
        # block that REPLACES whatever went wrong with a message about pointers. The
        # views this hands out (per record, and per message to on_message) are still
        # referenced by the traceback whenever a handler raises, so the useful error --
        # "chunk compressed with 'zstd', use the mcap CLI" on a foreign file -- was
        # arriving as "cannot close exported pointers exist". Closing is only about
        # releasing the mapping promptly; when it cannot, letting the GC do it costs
        # nothing and keeps the real exception.
        try:
            buf.close()
        except BufferError:
            pass
        fh.close()
    return s


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    path = argv[1]
    topic = None
    dump_dir = None
    limit = None
    i = 2
    while i < len(argv):
        if argv[i] == "--topic" and i + 1 < len(argv):
            topic = argv[i + 1]
            i += 2
        elif argv[i] == "--dump-dir" and i + 1 < len(argv):
            dump_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        else:
            print(f"unknown arg {argv[i]}")
            return 2

    printed = [0]
    dumped = [0]

    def on_msg(ch, log_t, pub_t, data):
        if topic is None or ch.topic != topic:
            return
        if limit is not None and printed[0] + dumped[0] >= limit:
            return
        if dump_dir is not None:
            import os
            os.makedirs(dump_dir, exist_ok=True)
            ext = {"png": ".png", "jpeg": ".jpg"}.get(ch.message_encoding, ".bin")
            name = f"{ch.topic.strip('/').replace('/', '_')}_{pub_t}{ext}"
            with open(os.path.join(dump_dir, name), "wb") as f:
                f.write(bytes(data))
            dumped[0] += 1
        elif ch.message_encoding == "json":
            print(json.dumps({"topic": ch.topic, "log_time": log_t,
                              "publish_time": pub_t,
                              "msg": json.loads(bytes(data))}))
            printed[0] += 1
        else:
            print(json.dumps({"topic": ch.topic, "log_time": log_t,
                              "publish_time": pub_t,
                              "bytes": len(data), "encoding": ch.message_encoding}))
            printed[0] += 1

    s = read_session(path, on_msg if (topic or dump_dir) else None)
    if topic is None and dump_dir is None:
        # Summary: re-walk counting (channels carry counts after a full pass).
        s = read_session(path, lambda *a: None)
        out = {
            "channels": [{"topic": c.topic, "encoding": c.message_encoding,
                          "schema": c.schema_name, "messages": c.message_count}
                         for c in sorted(s.channels.values(), key=lambda c: c.topic)],
            "metadata": s.metadata,
            "truncated": s.truncated,
        }
        print(json.dumps(out, indent=2))
    elif s.truncated:
        print(f"[mcap_reader] NOTE: {path} is truncated (recorder killed hard); "
              "messages up to the cut were read.", file=sys.stderr)
    if dumped[0]:
        print(f"[mcap_reader] dumped {dumped[0]} payload(s) to {dump_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
