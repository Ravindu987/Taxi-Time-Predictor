import { createTheme } from "@mui/material";

const theme = createTheme({
    palette: {
        primary: {
            main: "#b3b300",
        },
        secondary: {
            main: "#ffc61a",
        },
        background: {
            default: "#d9d9d9",
            paper: '#ffdf80', // your color
        },
    },
});

export default theme;