import React, { Suspense } from "react";
import AnnotateClient from "./AnnotateClient";

export default function Page() {
  return (
    <Suspense fallback={<main className="panel">Loading annotator…</main>}>
      <AnnotateClient />
    </Suspense>
  );
}

