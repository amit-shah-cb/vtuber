import React from "react";
import { MicrophoneMuteButton } from "./MicrophoneMuteButton";

const FloatingTopBar: React.FC = () => {
  return (
    <div className="fixed top-0 left-0 w-full z-50 bg-black bg-opacity-80 flex items-center justify-end px-4 py-2 shadow-lg">
      <MicrophoneMuteButton />
    </div>
  );
};

export default FloatingTopBar; 