import React from "react";

export interface ThoughtWindowProps {
  liveThoughts: string;
  chatHistory: { role: "user" | "assistant"; content: string }[];
  loading: boolean;
  phase: "planning" | "answering";
}

const ThoughtWindow: React.FC<ThoughtWindowProps> = ({
  liveThoughts,
  chatHistory,
  loading,
  phase,
}) => {
  return (
    <div className="bg-surface p-4 rounded shadow-glow text-white space-y-3">
      {phase === "planning" && loading && (
        <p className="italic">💭 {liveThoughts}</p>
      )}

      {chatHistory.length === 0 && (
        <p className="text-muted italic">No messages yet. Friday is listening.</p>
      )}
      {chatHistory.map((msg, i) => (
        <div key={i} className="text-sm">
          <span className="font-bold text-primary">{msg.role.toUpperCase()}:</span> {msg.content}
        </div>
      ))}

      {phase === "answering" && !loading && (
        <p className="italic">🤖 {liveThoughts}</p>
      )}
    </div>
  );
};

export default ThoughtWindow;
