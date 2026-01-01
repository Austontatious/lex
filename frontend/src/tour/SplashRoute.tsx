import React from "react";
import { Navigate, useNavigate } from "react-router-dom";
import SplashScreen from "../components/SplashScreen";
import { getUserId } from "../services/api";
import { setChatAutostart, shouldSkipSplash } from "./tour_storage";

export default function SplashRoute() {
  const navigate = useNavigate();
  const userId = getUserId();

  const skipSplash = shouldSkipSplash(userId);

  if (skipSplash) {
    setChatAutostart("direct");
    return <Navigate to="/chat" replace />;
  }

  return (
    <SplashScreen
      onTakeTour={() => navigate("/tour?force=1")}
      onTalkToLexi={() => {
        setChatAutostart("voice");
        navigate("/chat");
      }}
    />
  );
}
