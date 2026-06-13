import Nav from "./components/nav";
import "./globals.css";
import { Geist, Oswald } from "next/font/google";
import { cn } from "@/lib/utils";

const geist = Geist({subsets:['latin'],variable:'--font-sans'});
const oswald = Oswald({ subsets: ['latin'], weight: ['200','400','700'], variable: '--font-oswald' });


export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
  <html className={cn("font-sans", geist.variable, oswald.variable)}>
      <body  className="relative before:absolute before:top-0 before:left-0 before:w-full
      before:h-full before:content-[''] before:opacity-[0.025] before:z-10 before:pointer-events-none
      before:bg-[url('https://www.ui-layouts.com/noise.gif')]">
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}