import React, { useState } from 'react';
import { Code2, MessageSquare } from 'lucide-react';
import Sidebar from './Sidebar';
import ChatPane from './ChatPane';
import CodePane from './CodePane';
import SettingsModal from './SettingsModal';
import ModelPicker from './ModelPicker';
import ModelLibrary from './ModelLibrary';
import TitleBar from './TitleBar';
import { useStore } from '../state/store';

export default function Layout() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { tab, setTab } = useStore();

  return (
    <div className="h-screen w-screen flex flex-col bg-app text-[var(--fg)] overflow-hidden">
      <TitleBar />
      <div className="flex-1 flex min-h-0">
        <Sidebar onOpenSettings={() => setSettingsOpen(true)} />

      <main className="flex-1 flex flex-col min-w-0">
        <div className="h-12 flex items-center px-3 select-none gap-1 border-b bd-soft">
          <TabButton
            active={tab === 'chat'}
            onClick={() => setTab('chat')}
            label="Chat"
            icon={<MessageSquare className="w-3.5 h-3.5" />}
          />
          <TabButton
            active={tab === 'code'}
            onClick={() => setTab('code')}
            label="Code"
            icon={<Code2 className="w-3.5 h-3.5" />}
          />
          <div className="ml-auto">
            <ModelPicker />
          </div>
        </div>

        <div className="flex-1 min-h-0">
          {tab === 'chat' ? <ChatPane /> : <CodePane />}
        </div>
      </main>
      </div>

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
      <ModelLibrary />
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  icon: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[13px] transition ${
        active
          ? 'bg-[var(--bg-hover)] text-[var(--fg)]'
          : 'text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}
