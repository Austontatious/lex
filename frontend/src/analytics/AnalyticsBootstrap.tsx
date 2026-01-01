import { useEffect } from "react";
import { startAnalyticsTracking } from "./analyticsClient";

export default function AnalyticsBootstrap() {
  useEffect(() => startAnalyticsTracking(), []);
  return null;
}
