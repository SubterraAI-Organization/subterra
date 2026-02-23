import type { Metadata } from "next";

import "./globals.css";
import { Footer } from "./_components/Footer";
import { SideNav } from "./_components/SideNav";
import { TopNav } from "./_components/TopNav";

export const metadata: Metadata = {
  title: "Subterra HIL Annotator",
  description: "Human-in-the-loop mask correction for Subterra root segmentation"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="pageRoot">
          <TopNav />
          <div className="pageBody">
            <div className="container">
              <div className="appShell">
                <SideNav />
                <div style={{ minWidth: 0 }}>{children}</div>
              </div>
            </div>
          </div>
          <Footer />
        </div>
      </body>
    </html>
  );
}
