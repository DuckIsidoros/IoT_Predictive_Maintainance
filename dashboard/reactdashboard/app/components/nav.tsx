"use client";
import React from "react";
import Link from "next/link";

const Nav: React.FC = () => {

    const navbarRef = React.useRef<HTMLElement>(null);

  return (
  <header className="bg-white fixed top-0 left-0 w-full z-50 border-0 h-16">
      <nav
        ref={navbarRef}
        id="navbar"
        className="flex justify-between items-center px-5 h-full"
      >
        {/* Title and icon */}
        <div className="flex items-center space-x-2">
          <Link
            className="text-4xl text-[clamp(1.5rem,4vw,3rem)] font-bold text-white"
            href="../"
          >
            Galileo lens
          </Link>
        </div>

        <div className="space-4 flex items-center">
          <ul className="flex space-x-6">
            <li className="nav-underline-animate">
              <Link
                className="nav-text-hover inline-block text-base text-[clamp(1rem,1.5vw,1.25rem)] text-slate-300 duration-300 ease-in-out"
                href="../earth"
              >
                Home
              </Link>
            </li>
            <li className="nav-underline-animate">
              <Link
                className="nav-text-hover inline-block text-base text-[clamp(1rem,1.5vw,1.25rem)] text-slate-300 duration-300 ease-in-out"
                href="../luna"
              >
                About us
              </Link>
            </li>
            <li className="nav-underline-animate">
              <Link
                className="nav-text-hover inline-block text-base text-[clamp(1rem,1.5vw,1.25rem)] text-slate-300 duration-300 ease-in-out"
                href="../satellite"
              >
                Products
              </Link>
            </li>
          </ul>
        </div>
      </nav>
    </header>
  );
};

export default Nav;
