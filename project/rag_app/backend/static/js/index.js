//
document.querySelector("button").addEventListener("click", ask);
async function ask() {
  const question = document.querySelector("#question").value;
  const response = await fetch("/api/question", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question }),
  });

  const answer = await response.json();
  document.querySelector("#answer").textContent = answer.message;
}

// 파일 업로드
document.querySelector("#uploadBtn").addEventListener("click", uploadFile);
async function uploadFile() {
  const fileInput = document.querySelector("#file");
  const file = fileInput.files[0];

  if (!file) {
    alert("파일을 선택하세요");
    return;
  }

  // form을 만들어 전송
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/rag/upload", {
    method: "POST",
    body: formData,
  });

  const answer = await response.json();
  document.querySelector("#result").textContent = answer.message;
}

//
// document.querySelector("#askBtn").addEventListener("click", rag_ask);
async function rag_ask() {
  const question = document.querySelector("#question").value;
  const response = await fetch("/rag/question/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question }),
  });

  const answer = await response.json();
  document.querySelector("#answer_result").textContent = answer.message;
}

//
document.querySelector("#askBtn").addEventListener("click", rag_ask_stream);
async function rag_ask_stream() {
  const question = document.querySelector("#question").value;
  const response = await fetch("/api/rag/question/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) break;

    answer += decoder.decode(value);
    document.querySelector("#answer_result").textContent = answer;
  }
}
