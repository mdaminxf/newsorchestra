import React from "react";
import styled from "styled-components";

const LoadingPage = () => {
  return (
    <StyledWrapper>
      <button className="button" data-text="Loading">
        <span className="actual-text">&nbsp;NewsOrchestra&nbsp;</span>
        <span aria-hidden="true" className="front-text">
          &nbsp;NewsOrchestra&nbsp;
        </span>
      </button>
    </StyledWrapper>
  );
};

const StyledWrapper = styled.div`
  /* Fullscreen centering */
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: #212121;

  .button {
    position: relative;
    border: none;
    background: transparent;
    --stroke-color: #ffffff7c;
    --ani-color: rgba(95, 3, 244, 0);
    --color-gar: linear-gradient(
      90deg,
      #03a9f4,
      #f441a5,
      #ffeb3b,
      #03a9f4
    );
    letter-spacing: 3px;
    font-size: 3em;
    font-family: "Arial";
    text-transform: uppercase;
    color: transparent;
    -webkit-text-stroke: 1px var(--stroke-color);
    cursor: default; /* no pointer for loading */
  }


  .front-text {
    position: absolute;
    top: 0;
    left: 0;
    width: 0%;
    background: var(--color-gar);
    -webkit-background-clip: text;
    background-clip: text;
    background-size: 200%;
    overflow: hidden;
    transition: all 1s;
    animation: 8s shimmer infinite;
    border-bottom: 2px solid transparent;
  }

  @keyframes shimmer {
    0% {
      background-position: 0%;
      }
      50% {
        width:100%;
      background-position: 100%;
    }
  }
`;

export default LoadingPage;
