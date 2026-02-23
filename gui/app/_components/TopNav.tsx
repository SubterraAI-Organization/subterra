"use client";

import Link from "next/link";

function LogoMark() {
  return (
    <svg width="32" height="32" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="24" height="24" rx="8" stroke="rgba(255,255,255,0.18)" />
      <path
        d="M17.7 8.8c-.6-1.3-2-2.1-3.8-2.1-2.4 0-4.1 1.3-4.1 3.3 0 1.8 1.3 2.7 3.6 3.2l.7.2c1.3.3 1.9.7 1.9 1.4 0 .8-.9 1.4-2.1 1.4-1.2 0-2.3-.5-2.9-1.5"
        stroke="rgba(154, 190, 255, 0.98)"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function UserIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M20 21c0-4.1-3.6-6-8-6s-8 1.9-8 6"
        stroke="rgba(233,237,255,0.9)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z"
        stroke="rgba(233,237,255,0.9)"
        strokeWidth="2"
      />
    </svg>
  );
}

export function TopNav() {
  return (
    <header className="topNav">
      <div className="topNavInner">
        <Link href="/" className="brand" aria-label="Subterra Home">
          <LogoMark />
          <div className="brandText">
            <div className="brandName">Subterra</div>
            <div className="brandTag">human-in-the-loop</div>
          </div>
        </Link>

        <button className="userButton" type="button" aria-label="User menu (coming soon)" title="User (coming soon)">
          <UserIcon />
        </button>
      </div>
    </header>
  );
}
