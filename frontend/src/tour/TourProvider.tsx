import React, { useMemo, useState, useEffect } from "react";
import Joyride, { CallBackProps, STATUS, Step } from "react-joyride";
import { useLocation, useNavigate } from "react-router-dom";

type Props = { children: React.ReactNode };

const ENABLE_BEACON_TOUR = false;

const steps: Step[] = [
  { target: '[data-tour="avatar"]', content: "Your persistent avatar lives here." },
  { target: '[data-tour="modes"]', content: "Switch Lexi modes or review persona status here." },
  { target: '[data-tour="composer"]', content: "Type messages here. Press Enter to send." },
  { target: '[data-tour="gallery"]', content: "Open the avatar gallery & regeneration tools." },
];

export const TourContext = React.createContext<{ start: () => void }>({ start: () => {} });

export default function TourProvider({ children }: Props) {
  const [run, setRun] = useState(false);
  const [key, setKey] = useState(0);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (!ENABLE_BEACON_TOUR) return;
    const params = new URLSearchParams(location.search);
    if (params.get("tour") === "start") {
      setRun(true);
      params.delete("tour");
      navigate({ pathname: location.pathname, search: params.toString() }, { replace: true });
    }
  }, [location, navigate]);

  const contextValue = useMemo(
    () => ({
      start: () => {
        if (!ENABLE_BEACON_TOUR) return;
        setKey((prev) => prev + 1);
        setRun(true);
      },
    }),
    []
  );

  const handleCallback = (data: CallBackProps) => {
    if (data.status === STATUS.FINISHED || data.status === STATUS.SKIPPED) {
      setRun(false);
    }
  };

  return (
    <TourContext.Provider value={contextValue}>
      {children}
      {ENABLE_BEACON_TOUR && (
        <Joyride
          key={key}
          steps={steps}
          run={run}
          continuous
          showSkipButton
          disableScrolling
          styles={{ options: { zIndex: 10_000 } }}
          callback={handleCallback}
        />
      )}
    </TourContext.Provider>
  );
}
