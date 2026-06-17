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

  const response = await fetch("/api/card/upload", {
    method: "POST",
    body: formData,
  });

  const answer = await response.json();
  document.querySelector("#result").textContent = answer.message;
}
