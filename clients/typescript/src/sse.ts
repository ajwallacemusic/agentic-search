/** A spec-following server-sent events parser over a byte stream (WHATWG HTML §9.2.6). */

export interface SseMessage {
  event: string;
  data: string;
  id: string | null;
}

/**
 * Yields each dispatched message. Handles `\n`, `\r\n` and `\r` line endings, fields split
 * across chunks, multi-line `data`, comments and a leading BOM. An unterminated message at the
 * end of the stream is discarded, as the spec requires.
 */
export async function* parseSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseMessage, void, undefined> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let first = true;
  let event = "";
  let data: string[] = [];
  let id: string | null = null;
  let sawData = false;
  let ended = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (first && buffer.length > 0) {
        if (buffer.charCodeAt(0) === 0xfeff) buffer = buffer.slice(1);
        first = false;
      }
      for (;;) {
        const match = /\r\n|\r|\n/.exec(buffer);
        // A trailing "\r" might be the first half of "\r\n": wait for more input unless done.
        if (!match || (match[0] === "\r" && match.index === buffer.length - 1 && !done)) break;
        const line = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        if (line === "") {
          if (sawData) yield { event: event || "message", data: data.join("\n"), id };
          event = "";
          data = [];
          sawData = false;
          continue;
        }
        if (line.startsWith(":")) continue;
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let text = colon === -1 ? "" : line.slice(colon + 1);
        if (text.startsWith(" ")) text = text.slice(1);
        if (field === "event") event = text;
        else if (field === "data") {
          data.push(text);
          sawData = true;
        } else if (field === "id" && !text.includes("\0")) id = text;
      }
      if (done) {
        ended = true;
        return;
      }
    }
  } finally {
    // Stopped early (consumer broke out or threw): cancel the body so the connection closes.
    if (!ended) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
