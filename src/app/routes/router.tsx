import { createHashRouter } from "react-router-dom";
import { App } from "../App";

export const appRouter = createHashRouter([
  {
    path: "*",
    element: <App />
  }
]);
