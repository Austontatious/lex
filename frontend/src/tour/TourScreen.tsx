import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { CardDeck, DeckCard } from "../components/tour/CardDeck";
import { getUserId, sendTourPrompt } from "../services/api";
import {
  setChatAutostart,
  setChatPrefill,
  shouldSkipSplash,
  shouldSkipTourCards,
} from "./tour_storage";
import { tourCards, TourCard } from "./tour_cards";
import "../styles/tour.css";

type DemoState = {
  loading: boolean;
  response?: string;
  error?: string;
};

export default function TourScreen() {
  const navigate = useNavigate();
  const location = useLocation();
  const userId = getUserId();
  const [activeIndex, setActiveIndex] = useState(0);
  const [demoState, setDemoState] = useState<Record<string, DemoState>>({});

  useEffect(() => {
    const forceTour = new URLSearchParams(location.search).get("force") === "1";
    if (forceTour) {
      setActiveIndex(0);
      return;
    }
    if (shouldSkipSplash(userId)) {
      setChatAutostart("direct");
      navigate("/chat", { replace: true });
      return;
    }
    if (shouldSkipTourCards(userId)) {
      navigate("/tour/legal", { replace: true });
    }
  }, [location.search, navigate, userId]);

  const runDemo = async (cardId: string, prompt: string) => {
    setDemoState((prev) => ({
      ...prev,
      [cardId]: { loading: true },
    }));
    try {
      const payload = await sendTourPrompt(prompt, cardId);
      const reply =
        typeof payload?.cleaned === "string"
          ? payload.cleaned
          : payload?.choices?.[0]?.text || "";
      setDemoState((prev) => ({
        ...prev,
        [cardId]: { loading: false, response: reply || "No response yet." },
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Couldn't reach Lexi right now.";
      setDemoState((prev) => ({
        ...prev,
        [cardId]: { loading: false, error: message },
      }));
    }
  };

  const renderCardContent = (card: TourCard, cardId: string) => {
    if (card.kind === "demo") {
      const state = demoState[cardId] || { loading: false };
      return (
        <div className="tour-card-copy">
          <p className="tour-muted">{card.body}</p>
          <div className="tour-bubble-stack">
            <div className="tour-bubble tour-bubble--prompt">{card.prompt}</div>
            <div className="tour-bubble tour-bubble--reply">
              {state.loading
                ? "Lexi is thinking..."
                : state.error
                ? state.error
                : state.response || "Tap \"Try it\" to see Lexi respond."}
            </div>
          </div>
          <div className="tour-actions">
            <button
              type="button"
              className="btn primary small"
              onClick={() => void runDemo(cardId, card.prompt)}
            >
              {card.ctaLabel}
            </button>
            {state.response && !state.loading && (
              <>
                <button
                  type="button"
                  className="tour-link"
                  onClick={() => void runDemo(cardId, card.prompt)}
                >
                  Regenerate
                </button>
                <button
                  type="button"
                  className="tour-link"
                  onClick={() => {
                    setChatPrefill("I've been thinking about ___ lately...");
                    setChatAutostart("voice");
                    navigate("/chat");
                  }}
                >
                  Ask Lexi something real
                </button>
              </>
            )}
          </div>
        </div>
      );
    }

    if (card.kind === "cta") {
      const secondaryTarget = card.secondaryTo || "/chat";
      return (
        <div className="tour-card-copy">
          <p className="tour-muted">{card.body}</p>
          <div className="tour-cta-grid">
            <button
              type="button"
              className="btn primary"
              onClick={() => navigate(card.primaryTo)}
            >
              {card.primaryLabel}
            </button>
            {card.secondaryLabel ? (
              <button
                type="button"
                className="btn ghost"
                onClick={() => {
                  setChatAutostart("voice");
                  navigate(secondaryTarget);
                }}
              >
                {card.secondaryLabel}
              </button>
            ) : null}
          </div>
        </div>
      );
    }

    return (
      <div className="tour-card-copy">
        <p className="tour-muted">{card.body}</p>
      </div>
    );
  };

  const cards: DeckCard[] = tourCards.map((card, index) => {
    const id = `tour-card-${index}`;
    const nextLabel = card.kind === "cta" ? "Continue" : "Next";
    const onNext =
      card.kind === "cta"
        ? () => {
            navigate(card.primaryTo);
            return false;
          }
        : undefined;
    return {
      id,
      title: card.title,
      content: renderCardContent(card, id),
      nextLabel,
      onNext,
    };
  });

  return (
    <div className="tour-root">
      <div className="tour-shell">
        <CardDeck cards={cards} activeIndex={activeIndex} onIndexChange={setActiveIndex} />
      </div>
    </div>
  );
}
