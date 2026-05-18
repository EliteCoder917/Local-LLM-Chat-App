import React, { useState } from 'react';
import {
  Plus, Search, MessageSquare, Settings, Trash2, Library,
  PanelLeftClose, PanelLeft, User,
} from 'lucide-react';
import { useStore } from '../state/store';

interface Props {
  onOpenSettings: () => void;
}

export default function Sidebar({ onOpenSettings }: Props) {
  const {
    conversations,
    activeId,
    newConversation,
    selectConversation,
    deleteConversation,
    renameConversation,
    setTab,
    setLibraryOpen,
  } = useStore();
  const [collapsed, setCollapsed] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [search, setSearch] = useState('');

  function startNew() {
    newConversation();
    setTab('chat');
  }

  if (collapsed) {
    return (
      <aside className="w-14 shrink-0 bg-side border-r bd-soft flex flex-col items-center py-2 gap-1">
        <IconBtn onClick={() => setCollapsed(false)} title="Expand sidebar" icon={<PanelLeft className="w-4 h-4" />} />
        <IconBtn onClick={startNew} title="New chat" icon={<Plus className="w-4 h-4" />} />
        <IconBtn onClick={() => setLibraryOpen(true)} title="Models" icon={<Library className="w-4 h-4" />} />
        <div className="flex-1" />
        <IconBtn onClick={onOpenSettings} title="Settings" icon={<Settings className="w-4 h-4" />} />
      </aside>
    );
  }

  const filtered = search.trim()
    ? conversations.filter((c) => c.title.toLowerCase().includes(search.toLowerCase()))
    : conversations;

  return (
    <aside className="w-64 shrink-0 bg-side flex flex-col border-r bd-soft">
      {/* top bar: collapse + new chat */}
      <div className="px-2 py-2 flex items-center gap-1">
        <IconBtn onClick={() => setCollapsed(true)} title="Collapse" icon={<PanelLeftClose className="w-4 h-4" />} />
        <button
          onClick={startNew}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[13px] text-[var(--fg)]"
          title="New chat"
        >
          <Plus className="w-4 h-4" /> New chat
        </button>
      </div>

      {/* search */}
      <div className="px-2">
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--bg-hover)] focus-within:bg-[var(--bg-hover)]">
          <Search className="w-3.5 h-3.5 text-[var(--fg-dim)]" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search chats"
            className="flex-1 bg-transparent outline-none text-[13px] placeholder:text-[var(--fg-dim)]"
          />
        </div>
      </div>

      {/* models shortcut */}
      <div className="px-2 pt-1">
        <button
          onClick={() => setLibraryOpen(true)}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[13px] text-[var(--fg)]"
        >
          <Library className="w-3.5 h-3.5" /> Models
        </button>
      </div>

      {/* recents */}
      <div className="px-3 pt-3 pb-1 text-[11px] uppercase tracking-wider text-[var(--fg-dim)]">
        Chats
      </div>
      <div className="flex-1 overflow-auto scroll px-2 pb-2">
        {filtered.length === 0 && (
          <div className="px-2 py-3 text-xs text-[var(--fg-dim)]">
            {search ? 'No matches.' : 'No conversations yet.'}
          </div>
        )}
        {filtered.map((c) => {
          const active = c.id === activeId;
          const editing = c.id === editingId;
          return (
            <div
              key={c.id}
              className={`group flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] cursor-pointer ${
                active ? 'bg-[var(--bg-hover)] text-[var(--fg)]' : 'text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]'
              }`}
              onClick={() => {
                if (!editing) { selectConversation(c.id); setTab('chat'); }
              }}
              onDoubleClick={() => { setEditingId(c.id); setDraft(c.title); }}
            >
              <MessageSquare className="w-3.5 h-3.5 shrink-0 text-[var(--fg-dim)]" />
              {editing ? (
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={() => { renameConversation(c.id, draft.trim() || 'New chat'); setEditingId(null); }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { renameConversation(c.id, draft.trim() || 'New chat'); setEditingId(null); }
                    else if (e.key === 'Escape') setEditingId(null);
                  }}
                  className="flex-1 bg-[var(--bg-input)] border bd-strong rounded px-1 outline-none text-[13px]"
                />
              ) : (
                <>
                  <span className="flex-1 truncate">{c.title}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteConversation(c.id); }}
                    className="opacity-0 group-hover:opacity-100 text-[var(--fg-dim)] hover:text-red-400 p-0.5"
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* footer: settings + profile */}
      <div className="border-t bd-soft px-2 py-2 space-y-0.5">
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[13px] text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          <Settings className="w-3.5 h-3.5" /> Settings
        </button>
        <div className="flex items-center gap-2 px-2 py-1.5 text-[13px] text-[var(--fg-muted)]">
          <div className="w-6 h-6 rounded-full bg-[var(--bg-hover)] flex items-center justify-center">
            <User className="w-3 h-3 text-[var(--fg-dim)]" />
          </div>
          <span className="truncate">Local</span>
        </div>
      </div>
    </aside>
  );
}

function IconBtn({
  onClick, title, icon,
}: { onClick: () => void; title: string; icon: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="p-1.5 rounded-lg text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]"
    >
      {icon}
    </button>
  );
}
