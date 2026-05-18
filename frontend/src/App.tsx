import React, { useEffect } from 'react';
import Layout from './components/Layout';
import { ThemeProvider } from './components/ThemeProvider';
import { useStore } from './state/store';
import { useChat } from './hooks/useChat';

export default function App() {
  const init = useStore((s) => s.init);
  useEffect(() => {
    init();
  }, [init]);

  // Subscribe to streaming LLM events globally.
  useChat();

  return (
    <ThemeProvider>
      <Layout />
    </ThemeProvider>
  );
}
