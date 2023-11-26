import React from "react";
import { Box } from "@mui/system";
import { Typography } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import theme from "../theme";

export function Header() {
  return (
    <ThemeProvider theme={theme}>
      <Box
        bgcolor="primary.main"
        p={2}
        m={3}
        borderRadius={3}
        sx={{ textAlign: "center" }}
      >
        <Typography variant="h4" gutterBottom m={0}>
          Taxi Time Predictor
        </Typography>
      </Box>
    </ThemeProvider>
  );
}

export default Header;
