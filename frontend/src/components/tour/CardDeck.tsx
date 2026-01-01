import React, { ReactNode, useEffect, useMemo, useRef } from "react";

export type DeckCard = {
  id: string;
  title?: string;
  eyebrow?: string;
  content: ReactNode;
  canProceed?: boolean;
  nextLabel?: string;
  backLabel?: string;
  onNext?: () => Promise<boolean> | boolean;
  onBack?: () => void;
  rightAction?: ReactNode;
};

type CardDeckProps = {
  cards: DeckCard[];
  activeIndex: number;
  onIndexChange: (index: number) => void;
  allowSwipe?: boolean;
  showProgress?: boolean;
};

const isFormElement = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
};

const isInteractiveElement = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return Boolean(
    target.closest(
      "button, a, input, textarea, select, [role='button'], [contenteditable='true']"
    )
  );
};

export const CardDeck: React.FC<CardDeckProps> = ({
  cards,
  activeIndex,
  onIndexChange,
  allowSwipe = true,
  showProgress = true,
}) => {
  const card = cards[activeIndex];
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const triggeredRef = useRef(false);

  const canProceed = card?.canProceed !== false;
  const hasNext = activeIndex < cards.length - 1;
  const nextEnabled = canProceed && (hasNext || Boolean(card?.onNext));
  const hasBack = activeIndex > 0 || Boolean(card?.onBack);

  const goNext = async () => {
    if (!card || !nextEnabled) {
      return;
    }
    let ok = true;
    if (card.onNext) {
      ok = await card.onNext();
    }
    if (!ok) {
      return;
    }
    if (hasNext) {
      onIndexChange(activeIndex + 1);
    }
  };

  const goBack = () => {
    if (!card || !hasBack) {
      return;
    }
    if (card.onBack) {
      card.onBack();
      return;
    }
    if (activeIndex > 0) {
      onIndexChange(activeIndex - 1);
    }
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.isComposing || isFormElement(event.target)) {
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        if (event.shiftKey) {
          goBack();
        } else {
          void goNext();
        }
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        void goNext();
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goBack();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!allowSwipe) {
      return;
    }
    if (isInteractiveElement(event.target)) {
      return;
    }
    startRef.current = { x: event.clientX, y: event.clientY };
    triggeredRef.current = false;
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!allowSwipe || !startRef.current || triggeredRef.current) {
      return;
    }
    const dx = event.clientX - startRef.current.x;
    const dy = event.clientY - startRef.current.y;
    if (Math.abs(dx) > 48 && Math.abs(dx) > Math.abs(dy) * 1.2) {
      triggeredRef.current = true;
      if (dx < 0) {
        void goNext();
      } else {
        goBack();
      }
    }
  };

  const handlePointerUp = () => {
    startRef.current = null;
    triggeredRef.current = false;
  };

  const stack = useMemo(() => {
    return cards.map((item, index) => {
      const offset = index - activeIndex;
      let position = "hidden";
      if (offset === 0) position = "active";
      if (offset === -1) position = "prev";
      if (offset === 1) position = "next";
      return { item, index, position, offset };
    });
  }, [cards, activeIndex]);

  if (!card) {
    return null;
  }

  return (
    <div className="card-deck">
      <div className="card-stack">
        {stack.map(({ item, index, position, offset }) => (
          <section
            key={item.id}
            className={`card card--${position}`}
            data-offset={offset}
            aria-hidden={position !== "active"}
          >
            <div className="card-shell">
              <div className="card-header">
                <div>
                  {showProgress && (
                    <div className="card-progress-row">
                      <div className="card-progress">
                        Step {activeIndex + 1} of {cards.length}
                      </div>
                      <div className="card-dots" aria-hidden="true">
                        {cards.map((_, dotIndex) => (
                          <span
                            key={`dot-${dotIndex}`}
                            className={`card-dot${dotIndex === activeIndex ? " is-active" : ""}`}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  {item.eyebrow && <div className="card-eyebrow">{item.eyebrow}</div>}
                  {item.title && <h2 className="card-title">{item.title}</h2>}
                </div>
                {item.rightAction && <div className="card-action">{item.rightAction}</div>}
              </div>

              <div
                className="card-body"
                onPointerDown={index === activeIndex ? handlePointerDown : undefined}
                onPointerMove={index === activeIndex ? handlePointerMove : undefined}
                onPointerUp={index === activeIndex ? handlePointerUp : undefined}
                onPointerCancel={index === activeIndex ? handlePointerUp : undefined}
              >
                {item.content}
              </div>

              <div className="card-footer">
                <button
                  type="button"
                  className="btn ghost"
                  onClick={goBack}
                  disabled={!hasBack}
                >
                  {item.backLabel || "Back"}
                </button>
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => void goNext()}
                  disabled={!nextEnabled}
                >
                  {item.nextLabel || "Next"}
                </button>
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
};
