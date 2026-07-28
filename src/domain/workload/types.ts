export type TokenSweepMode = "fixed-step";

export type WorkloadInput = {
  prefillTokenLength: number;
  decodeOutputTokens: number | null;
  tokenRangeStart: number;
  tokenRangeEnd: number;
  tokenRangeStep: number;
  tokenSweepMode: TokenSweepMode;
};
