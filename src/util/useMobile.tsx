import { useEffect, useState } from "react";

export const useMobile = () => {
  const [isMobile, setIsMobile] = useState(false);
  
  useEffect(() => {
    if (typeof window !== "undefined") {
      const userAgent = navigator.userAgent;
      const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
      const isMobileDevice = mobileRegex.test(userAgent);
      setIsMobile(isMobileDevice);
      
      // No need for resize listener since user agent doesn't change
      // Device type is determined once and stays constant
    }
  }, []);
  
  return isMobile;
};
