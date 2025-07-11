import { MicrophoneMuteButton } from "./MicrophoneMuteButton";
import { MicrophoneSelect } from "./MicrophoneSelect";
import { WebCamSelect } from "./WebCamSelect";

export function FloatingTopBar() {
  return (
    <div className="fixed top-0 left-0 right-0 bg-black/80 backdrop-blur-sm border-b border-gray-700 p-4 z-50">
      <div className="flex items-center justify-between gap-4 max-w-6xl mx-auto">
        <div className="flex items-center gap-4">
          <MicrophoneMuteButton />
          <MicrophoneSelect />
          <WebCamSelect />
        </div>
        
        <div className="flex items-center gap-4">
          {/* Additional controls can be added here */}
        </div>
      </div>
    </div>
  );
} 