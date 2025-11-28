'use client';

import React, { useState } from 'react';
import { RoomAudioRenderer, StartAudio } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { SessionProvider } from '@/components/app/session-provider';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/livekit/toaster';

import { JoinForm } from '@/components/app/JoinForm';

interface AppProps {
  appConfig: AppConfig;
}

interface ConnDetails {
  serverUrl: string;
  participantToken: string;
  roomName: string;
  participantName: string;
}

export function App({ appConfig }: AppProps) {
  const [connDetails, setConnDetails] = useState<ConnDetails | null>(null);

  const handleJoin = async (room: string, user: string) => {
    try {
      const response = await fetch('/api/connection-details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          roomName: room,
          participantName: user,
        }),
      });
      if (!response.ok) {
        throw new Error('Failed to get connection details');
      }
      const data = await response.json();
      console.log('Connection data:', data);

      setConnDetails({
        serverUrl: data.serverUrl,
        participantToken: data.participantToken,
        roomName: data.roomName,
        participantName: data.participantName,
      });
    } catch (error) {
      console.error(error);
      alert('Error joining room: ' + (error instanceof Error ? error.message : String(error)));
    }
  };

  // If not joined yet (no connection details), show the form
  if (!connDetails) {
    return <JoinForm onJoin={handleJoin} />;
  }

  // Once joined, render session
  const updatedConfig: AppConfig = {
    ...appConfig,
    roomName: connDetails.roomName,
    participantName: connDetails.participantName,
    serverUrl: connDetails.serverUrl,
    participantToken: connDetails.participantToken,
  };

  return (
    <SessionProvider appConfig={updatedConfig}>
      <main className="grid h-svh grid-cols-1 place-content-center">
        <ViewController />
      </main>
      <StartAudio label="Start Audio" />
      <RoomAudioRenderer />
      <Toaster />
    </SessionProvider>
  );
}


// 'use client';

// import { RoomAudioRenderer, StartAudio } from '@livekit/components-react';
// import type { AppConfig } from '@/app-config';
// import { SessionProvider } from '@/components/app/session-provider';
// import { ViewController } from '@/components/app/view-controller';
// import { Toaster } from '@/components/livekit/toaster';

// interface AppProps {
//   appConfig: AppConfig;
// }

// export function App({ appConfig }: AppProps) {
//   return (
//     <SessionProvider appConfig={appConfig}>
//       <main className="grid h-svh grid-cols-1 place-content-center">
//         <ViewController />
//       </main>
//       <StartAudio label="Start Audio" />
//       <RoomAudioRenderer />
//       <Toaster />
//     </SessionProvider>
//   );
// }
