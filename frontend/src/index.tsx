import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider } from "@chakra-ui/react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./index.css";
import App from "./App";
import AnalyticsDashboard from "./analytics/AnalyticsDashboard";
import AnalyticsBootstrap from "./analytics/AnalyticsBootstrap";
import SplashRoute from "./tour/SplashRoute";
import TourLegal from "./tour/TourLegal";
import TourScreen from "./tour/TourScreen";
import TourProvider from "./tour/TourProvider";
import reportWebVitals from "./reportWebVitals";

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <ChakraProvider>
      <AnalyticsBootstrap />
      <BrowserRouter>
        <TourProvider>
          <Routes>
            <Route path="/" element={<SplashRoute />} />
            <Route path="/tour" element={<TourScreen />} />
            <Route path="/tour/legal" element={<TourLegal />} />
            <Route path="/chat" element={<App />} />
            <Route path="/analytics" element={<AnalyticsDashboard />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </TourProvider>
      </BrowserRouter>
    </ChakraProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
