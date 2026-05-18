import React from 'react';

interface Props {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
}

export default function Toggle({ checked, onChange, disabled }: Props) {
  return (
    <label className={`toggle inline-flex items-center ${disabled ? 'opacity-40 pointer-events-none' : 'cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
      />
      <span className="track" />
    </label>
  );
}
