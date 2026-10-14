/**
 * What a session journal line contributes to the history index. Encrypted
 * reasoning, images and metadata entries are not searchable text; they stay
 * in the journal untouched.
 */

export type EntryKind = 'user' | 'assistant' | 'tool_result' | 'bash' | 'custom' | 'compaction' | 'branch_summary';
export const ENTRY_KINDS: readonly EntryKind[] = ['user', 'assistant', 'tool_result', 'bash', 'custom', 'compaction', 'branch_summary'];

export type SessionHeader = Readonly<{ sessionId: string; cwd: string; startedAt: string }>;

export type HistoryEntry = Readonly<{
  entryId: string;
  parentId: string | null;
  timestamp: string;
  kind: EntryKind;
  /** Message role, or the custom message type. */
  role: string;
  /** Tool names (assistant calls, tool results) or the custom type. */
  name: string | null;
  text: string;
  tokensBefore: number | null;
}>;

type Block = { type?: string; text?: unknown; name?: unknown; arguments?: unknown };

const blocks = (content: unknown): Block[] => Array.isArray(content) ? content.filter(block => block && typeof block === 'object') as Block[] : [];

/** Visible text: text blocks, tool calls as `name(args)`, an image marker. Reasoning is skipped. */
export const contentText = (content: unknown): string => typeof content === 'string' ? content : blocks(content).map(block => {
  if (block.type === 'text') return String(block.text ?? '');
  if (block.type === 'toolCall') return `${String(block.name ?? 'tool')}(${JSON.stringify(block.arguments ?? {})})`;
  if (block.type === 'image') return '[image]';
  return '';
}).filter(Boolean).join('\n');

const text = (value: unknown): string => typeof value === 'string' ? value : '';

export const headerOf = (line: Record<string, unknown>): SessionHeader | undefined => line.type === 'session' && typeof line.id === 'string'
  ? { sessionId: line.id, cwd: text(line.cwd), startedAt: text(line.timestamp) } : undefined;

export const entryOf = (line: Record<string, unknown>): HistoryEntry | undefined => {
  if (typeof line.id !== 'string') return undefined;
  const base = { entryId: line.id, parentId: typeof line.parentId === 'string' ? line.parentId : null, timestamp: text(line.timestamp), tokensBefore: null };
  if (line.type === 'compaction' || line.type === 'branch_summary') {
    const summary = text(line.summary);
    return summary ? { ...base, kind: line.type, role: line.type, name: null, text: summary,
      tokensBefore: typeof line.tokensBefore === 'number' ? line.tokensBefore : null } : undefined;
  }
  if (line.type === 'custom_message') {
    const body = contentText(line.content);
    const type = text(line.customType) || 'custom';
    return body ? { ...base, kind: 'custom', role: type, name: type, text: body } : undefined;
  }
  if (line.type !== 'message' || !line.message || typeof line.message !== 'object') return undefined;
  const message = line.message as Record<string, unknown>;
  const role = text(message.role);
  if (role === 'user') {
    const body = contentText(message.content).trim();
    return body ? { ...base, kind: 'user', role, name: null, text: body } : undefined;
  }
  if (role === 'assistant') {
    const body = contentText(message.content).trim();
    const tools = blocks(message.content).filter(block => block.type === 'toolCall').map(block => String(block.name ?? ''));
    return body ? { ...base, kind: 'assistant', role, name: tools.length ? [...new Set(tools)].join(',') : null, text: body } : undefined;
  }
  if (role === 'toolResult') {
    const body = contentText(message.content);
    return body ? { ...base, kind: 'tool_result', role, name: text(message.toolName) || null, text: body } : undefined;
  }
  if (role === 'bashExecution') {
    const body = [text(message.command) && `$ ${text(message.command)}`, text(message.output)].filter(Boolean).join('\n');
    return body ? { ...base, kind: 'bash', role, name: 'bash', text: body } : undefined;
  }
  return undefined;
};
