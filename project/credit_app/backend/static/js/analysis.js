document.querySelector("button").addEventListener("click", ask);

async function ask() {
  const question = document.querySelector("#question").value;

  const response = await fetch("/api/card/analysis", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question }),
  });

  const answer = await response.json();
  document.querySelector("#result").textContent = answer.message;
}
