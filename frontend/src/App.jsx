import { useState } from "react";
import axios from "axios";
import "./App.css";
function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const generateResponse = async () => {
    const res = await axios.post(
    "fintech-finetuned-llm-production.up.railway.app:8000/generate",
    {
      prompt: prompt
    }
    );
    setResponse(res.data.response);
  };
  
  return (
    <div className="container">
      
      <h1>Domain Specific AI Assistant</h1>
      
      <textarea
        placeholder="Ask Finance or Tech Questions..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      
      <button onClick={generateResponse}>
        Generate
      </button>
      
      <div className="response">
        {response}
      </div>
    
    </div>
  );
}

export default App;