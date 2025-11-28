'use client';

import React, { FormEvent, useState } from 'react';

interface JoinFormProps {
  onJoin: (roomName: string, userName: string) => void;
}   

export function JoinForm({ onJoin }: JoinFormProps) {
  const [roomName, setRoomName] = useState('');
  const [userName, setUserName] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!roomName || !userName) {
      alert('Please enter both room name and user name');
      return;
    }
    onJoin(roomName.trim(), userName.trim());
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-950 text-white">
      <form
        onSubmit={handleSubmit}
        className="flex w-80 flex-col gap-4 rounded-2xl bg-gray-900 p-8 shadow-lg"
      >
        <h1 className="text-center text-xl font-semibold">Join a Room</h1>

        <input
          type="text"
          placeholder="Room Name"
          value={roomName}
          onChange={(e) => setRoomName(e.target.value)}
          className="rounded border border-gray-700 bg-gray-800 p-2 outline-none focus:border-blue-500"
        />

        <input
          type="text"
          placeholder="Your Name"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          className="rounded border border-gray-700 bg-gray-800 p-2 outline-none focus:border-blue-500"
        />

        <button type="submit" className="rounded bg-blue-600 p-2 transition hover:bg-blue-700">
          Join Room
        </button>
      </form>
    </div>
  );
}
