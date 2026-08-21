import './BackendLoader.css'

// Пока бэкенд ещё не ответил ни разу (make run_web поднимает uvicorn и
// vite конкурентно, без синхронизации — холодный старт FastAPI занимает
// реальные секунды, см. App.tsx) — на пустой инпут можно успеть напечатать
// и отправить запрос, который тут же упадёт connection refused. Полноэкранный
// лоадер вместо этого — ничего не рисуем, пока реально нечего показывать.
export function BackendLoader() {
  return (
    <div className="backend-loader">
      <span className="brand-mark brand-mark-lg backend-loader-spin" aria-hidden="true">
        F
      </span>
      <p>Бэкенд запускается…</p>
    </div>
  )
}
