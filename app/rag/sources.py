def attachment_source_prefix(attachment_id: str) -> str:
    return f"attachment:{attachment_id}"


def attachment_remove_prefixes(attachment_ids: list[str]) -> list[str]:
    return [attachment_source_prefix(aid) for aid in attachment_ids]


def source_prefixes_from_attachment_ids(ids: list[str] | None) -> tuple[str, ...] | None:
    return tuple(attachment_source_prefix(i) for i in ids) if ids else None


def matches_source_prefix(source: str | None, prefixes: tuple[str, ...] | list[str]) -> bool:
    return bool(source) and source.startswith(tuple(prefixes))
