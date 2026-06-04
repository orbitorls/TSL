/** Strip TSL51 sign_id suffix to a readable Thai base word for UI. */
export function glossLabel(signId: string): string {
  if (!signId || signId.startsWith("Unknown /")) return signId;
  const idx = signId.indexOf("_");
  return idx > 0 ? signId.slice(0, idx) : signId;
}
