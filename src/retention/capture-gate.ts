/** Shared admission for agent capture and daemon publication; never a fail-open volume gate. */

export type CaptureAdmission = Readonly<{ accept: boolean; reason: string }>;

export const isNegationOrCorrection = (kind: string, statement: string): boolean =>
  kind === 'correction' || kind === 'constraint'
  || /\b(not|never|wrong|incorrect|false|instead|do not|don't)\b/iu.test(statement);

export const admitCapture = (input: Readonly<{
  statement: string;
  kind: string;
  existing: readonly Readonly<{ statement: string; kind: string }>[];
}>): CaptureAdmission => {
  const statement = input.statement.trim();
  if (!statement) return { accept: false, reason: 'empty' };
  if (input.existing.some(item => item.statement === statement)) return { accept: false, reason: 'duplicate' };
  if (isNegationOrCorrection(input.kind, statement)) return { accept: true, reason: 'correction' };
  return { accept: true, reason: 'novel' };
};
