export {};

declare global {
  interface Window {
    __subterraHandoff?: Record<
      string,
      {
        items: Array<{
          filename: string;
          originalDataUrl: string;
          maskDataUrl: string;
          metrics?: Record<string, any> | null;
          meta?: Record<string, any> | null;
        }>;
        createdAt: string;
        source: string;
      }
    >;
  }
}
