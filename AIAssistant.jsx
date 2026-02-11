import React, { useState } from "react";
import axios from "axios";
import "./ai.css";

export default function AIAssistant({ lat, lon, live }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text:
        "Hi! I am your AI Weather Assistant 🌦️ Ask anything about the weather!"
    }
  ]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userText = input;
    setInput("");
    setMessages((prev) => [...prev, { sender: "user", text: userText }]);

    try {
      const prompt = `
Location: ${lat ?? "N/A"}, ${lon ?? "N/A"}
Humidity: ${live?.humidity ?? "N/A"}
Wind: ${live?.wind ?? "N/A"}
Weather Description: ${live?.description ?? "Unknown"}
Question from user: ${userText}

You are an expert Tamil Nadu weather assistant.
Give a short, clear and helpful answer (2–4 sentences).
`.trim();

      const res = await axios.post("http://localhost:11434/api/generate", {
        model: "llama3.2",
        prompt,
        stream: false,   // IMPORTANT for single JSON response
      });

      console.log("OLLAMA RAW RESPONSE:", res.data);

      const reply = res.data?.response || "I couldn't process that.";
      setMessages((prev) => [...prev, { sender: "ai", text: reply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "AI offline. Start Ollama & try again." }
      ]);
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button className="ai-btn" onClick={() => setOpen(true)}>
        🤖
      </button>

      {/* Chat UI */}
      {open && (
        <div className="ai-box">
          <div className="ai-header">
            <b>AI Weather Assistant</b>
            <button className="close-btn" onClick={() => setOpen(false)}>✖</button>
          </div>

          <div className="ai-chat">
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.sender}`}>
                {m.text}
              </div>
            ))}
          </div>

          <div className="ai-input">
            <input
              placeholder="Ask me anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      )}
    </>
  );
}
