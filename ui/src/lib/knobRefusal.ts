/**
 * The hint of a B0-28 knob refusal.
 *
 * The backend refuses a knob with `{ detail: { code, hint, ... } }`; the hint
 * says what to do, so screens show it instead of a generic failure message.
 * Returns null when the error has no such hint, or when `code` is given and the
 * refusal carries a different code.
 */
export function refusalHint(err: unknown, code?: string): string | null {
    const detail = (err as { detail?: { code?: unknown; hint?: unknown } } | undefined)?.detail;
    if (code !== undefined && detail?.code !== code) return null;
    return typeof detail?.hint === "string" ? detail.hint : null;
}
