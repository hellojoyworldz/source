const API_URI = "http://localhost:11434/api/chat";

async function logJSONData() {
  const response = await fetch(API_URI, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "qwen2.5:7b",
      messages: [
        { role: "system", content: "당신은 친절한 AI 어시스턴스입니다." },
        { role: "user", content: "파이썬의 장점을 3가지 알려주세요" },
      ],
      stream: false,
    }),
  });
  const jsonData = await response.json();
  console.log(jsonData);
}

logJSONData();
