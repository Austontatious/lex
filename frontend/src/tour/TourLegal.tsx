import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiDisclaimerAck, fetchTourLegal, getUserId } from "../services/api";
import {
  markLegalAck,
  markTourCompleted,
  setChatAutostart,
  setDontShowAgain,
} from "./tour_storage";
import "../styles/tour.css";

export default function TourLegal() {
  const navigate = useNavigate();
  const [legalText, setLegalText] = useState("");
  const [loading, setLoading] = useState(true);
  const [showFull, setShowFull] = useState(false);
  const [dontShow, setDontShow] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchTourLegal()
      .then((res) => {
        if (!mounted) return;
        setLegalText(res?.text || "");
      })
      .catch(() => {
        if (!mounted) return;
        setLegalText("");
      })
      .finally(() => {
        if (mounted) {
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleFinish = async () => {
    if (submitting) return;
    setSubmitting(true);
    const userId = getUserId();
    markTourCompleted(userId);
    markLegalAck(userId);
    setDontShowAgain(dontShow, userId);
    setChatAutostart("direct");
    if (userId) {
      try {
        await apiDisclaimerAck(userId, true, "legal_v1");
      } catch {
        // still allow client-side completion if the ack fails
      }
    }
    setSubmitting(false);
    navigate("/chat");
  };

  return (
    <div className="tour-legal">
      <div className="tour-legal-card">
        <h1 className="tour-legal-title">One quick thing</h1>
        <p className="tour-legal-summary">
          Lexi is an AI companion, not a replacement for professional support, and she may be
          imperfect. Please avoid sharing personal identifiers or sensitive details.
        </p>
        <button
          type="button"
          className="tour-legal-toggle"
          onClick={() => setShowFull((prev) => !prev)}
          aria-expanded={showFull}
        >
          {showFull ? "Hide the full legal stuff" : "Show the full legal stuff"}
        </button>
        {showFull && (
          <div className="tour-legal-details">
            {loading ? "Loading..." : legalText || "Legal text is unavailable right now."}
          </div>
        )}
        <label className="tour-checkbox">
          <input
            type="checkbox"
            checked={dontShow}
            onChange={(event) => setDontShow(event.target.checked)}
          />
          Don't show this again
        </label>
        <div className="tour-legal-actions">
          <button type="button" className="btn ghost" onClick={() => navigate("/tour")}>
            Back
          </button>
          <button type="button" className="btn primary" onClick={() => void handleFinish()}>
            {submitting ? "Finishing..." : "Finish"}
          </button>
        </div>
      </div>
    </div>
  );
}
