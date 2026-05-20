import React, { useEffect } from 'react';
import Layout from './components/Layout';
import PermissionModal from './components/PermissionModal';
import { ThemeProvider } from './components/ThemeProvider';
import { useStore } from './state/store';
import { useChat } from './hooks/useChat';
import { api, type PermissionRequest } from './ipc/bridge';

export default function App() {
  const init = useStore((s) => s.init);
  const setPermissionRequest = useStore((s) => s.setPermissionRequest);
  useEffect(() => {
    init();
  }, [init]);

  // Backend permission requests → custom in-app modal (see PermissionModal).
  useEffect(() => {
    return api.perms.onRequest((req) => setPermissionRequest(req as PermissionRequest));
  }, [setPermissionRequest]);

  // Subscribe to streaming LLM events globally.
  useChat();

  return (
    <ThemeProvider>
      <Layout />
      <PermissionModal />
    </ThemeProvider>
  );
}
