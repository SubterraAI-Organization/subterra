"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Dashboard" },
  { href: "/annotate", label: "Annotation" },
  { href: "/retrain", label: "Re-training" },
  { href: "/phenotype", label: "Phenotyping" },
  { href: "/genotype", label: "Genotyping" },
  { href: "/api", label: "API" },
  { href: "/observability", label: "Observability" }
];

export function SideNav() {
  const pathname = usePathname();
  return (
    <nav
      style={{
        display: "grid",
        gap: 8,
        padding: 12,
        borderRadius: 14,
        border: "1px solid var(--border)",
        background: "rgba(0, 0, 0, 0.14)"
      }}
    >
      <div style={{ padding: "6px 8px", color: "var(--muted)", fontSize: 12 }}>Subterra</div>
      {items.map((it) => {
        const active = pathname === it.href;
        return (
          <Link
            key={it.href}
            href={it.href}
            style={{
              textDecoration: "none",
              padding: "10px 10px",
              borderRadius: 12,
              border: active ? "1px solid rgba(122, 162, 255, 0.5)" : "1px solid transparent",
              background: active ? "rgba(122, 162, 255, 0.12)" : "transparent"
            }}
          >
            {it.label}
          </Link>
        );
      })}
    </nav>
  );
}
