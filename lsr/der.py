from __future__ import annotations

from .errors import RouterError


def der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def der_tlv(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + der_length(len(payload)) + payload


def der_sequence(*items: bytes) -> bytes:
    return der_tlv(0x30, b"".join(items))


def der_integer(value: int) -> bytes:
    if value < 0:
        raise RouterError("DER encoder only supports non-negative integers.")
    raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return der_tlv(0x02, raw)


def der_octet_string(payload: bytes) -> bytes:
    return der_tlv(0x04, payload)


class DerReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def read_tlv(self, expected_tag: int | None = None) -> tuple[int, bytes]:
        if self.offset >= len(self.data):
            raise RouterError("Unexpected end of DER data.")
        tag = self.data[self.offset]
        self.offset += 1
        if expected_tag is not None and tag != expected_tag:
            raise RouterError(f"Unexpected DER tag 0x{tag:02x}; expected 0x{expected_tag:02x}.")

        if self.offset >= len(self.data):
            raise RouterError("Missing DER length.")
        first = self.data[self.offset]
        self.offset += 1
        if first < 0x80:
            length = first
        else:
            n_bytes = first & 0x7F
            if n_bytes == 0 or self.offset + n_bytes > len(self.data):
                raise RouterError("Invalid DER length.")
            length = int.from_bytes(self.data[self.offset : self.offset + n_bytes], "big")
            self.offset += n_bytes

        if self.offset + length > len(self.data):
            raise RouterError("DER payload length exceeds file size.")
        payload = self.data[self.offset : self.offset + length]
        self.offset += length
        return tag, payload

    def read_sequence(self) -> "DerReader":
        _, payload = self.read_tlv(0x30)
        return DerReader(payload)

    def read_integer(self) -> int:
        _, payload = self.read_tlv(0x02)
        if not payload:
            raise RouterError("Empty DER INTEGER.")
        return int.from_bytes(payload, "big", signed=False)

    def read_octet_string(self) -> bytes:
        _, payload = self.read_tlv(0x04)
        return payload
