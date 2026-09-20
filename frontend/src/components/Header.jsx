export default function Header() {
  return (
    <header className="header">
      <span className="header__mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 32 32" fill="none">
          <path
            d="M5 11V6h5M27 11V6h-5M5 21v5h5M27 21v5h-5"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
          />
        </svg>
      </span>
      <span className="header__name">Scanline</span>
      <span className="header__tagline">MRI tumor screening</span>
    </header>
  );
}
