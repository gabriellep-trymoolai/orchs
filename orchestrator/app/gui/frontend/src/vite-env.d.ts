/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_API_VERSION: string;
  readonly VITE_WS_BASE_URL: string;
  readonly VITE_WS_ENDPOINT: string;
  readonly VITE_ORGANIZATION_ID: string;
  readonly VITE_APP_ENV: string;
  readonly VITE_DEBUG: string;
  readonly VITE_ENABLE_CACHE: string;
  readonly VITE_ENABLE_FIREWALL: string;
  readonly VITE_ENABLE_MONITORING: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
