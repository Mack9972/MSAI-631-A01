import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [text, setText] = useState("");
  const [includeEnglish, setIncludeEnglish] = useState(true);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    const prompt = text.trim();
    if (!prompt) {
      setError("Please enter an English prompt.");
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: prompt,
    };
    setMessages((prev) => [...prev, userMessage]);
    setText("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt, include_english: includeEnglish }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed.");
      }

      const data = await response.json();
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        english: data.english || null,
        spanish: data.spanish || "",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <header className="header">
          <p className="eyebrow">FastAPI + React</p>
          <h1>Messaging Chatbot UI</h1>
          <p className="subtitle">
            Send messages in sequence and get replies like a chat app.
          </p>
        </header>

        <section className="chat-window" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty-chat">
              Start with a question, for example: "What is a chatbot?"
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <article key={message.id} className="message message-user">
                <p>{message.text}</p>
              </article>
            ) : (
              <article key={message.id} className="message message-assistant">
                {message.english && (
                  <>
                    <h2>English</h2>
                    <p>{message.english}</p>
                  </>
                )}
                <h2>Spanish</h2>
                <p>{message.spanish}</p>
              </article>
            ),
          )}

          {loading && <div className="typing">Assistant is replying...</div>}
          <div ref={endRef} />
        </section>

        <form className="form" onSubmit={handleSubmit}>
          <div className="controls">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={includeEnglish}
                onChange={(event) => setIncludeEnglish(event.target.checked)}
              />
              Include English response
            </label>
          </div>

          <div className="composer">
            <textarea
              id="prompt"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type your message..."
              rows={2}
            />
            <button type="submit" disabled={loading}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}
