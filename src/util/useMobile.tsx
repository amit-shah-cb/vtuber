import { useEffect, useState } from "react";

export const useMobile = () => {
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    if (typeof window !== "undefined") {
      const checkMobile = () => {
        const userAgent = navigator.userAgent;
        const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
        setIsMobile(mobileRegex.test(userAgent));
      };
      
      checkMobile();
      
      // Listen for resize events in case orientation changes
      window.addEventListener('resize', checkMobile);
      return () => window.removeEventListener('resize', checkMobile);
    }
  }, []);
  
  return isMobile;
};
