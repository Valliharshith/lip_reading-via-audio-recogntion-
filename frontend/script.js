async function send(){
  const file = document.getElementById("file").files[0]

  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData
  })

  const data = await res.json()

  document.getElementById("result").textContent =
    JSON.stringify(data, null, 2)
}
