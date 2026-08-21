import './VoiceOrb.css'

// Пульсирующий "шарик" войс-режима — тот же градиент/цвет, что и
// brand-mark (кубик в сайдбаре), просто круглый и "бурлящий" (border-radius
// анимация вместо простого масштаба). Показывается, пока идёт запись,
// пока ждём ответ, и пока играет TTS-озвучка ответа — label меняется,
// сам шарик — нет, чтобы состояние читалось по подписи, а не по виду.
export function VoiceOrb({ label }: { label?: string }) {
  return (
    <div className="voice-orb-wrap">
      <div className="voice-orb" aria-hidden="true">
        <span className="voice-orb-ring-b" />
        <span className="voice-orb-ring-a" />
        <span className="voice-orb-core" />
      </div>
      {label && <div className="voice-orb-label">{label}</div>}
    </div>
  )
}
