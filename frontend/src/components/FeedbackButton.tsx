import { useState } from "react";
import { Button } from "@chakra-ui/react";

import { FeedbackModal } from "./FeedbackModal";

export function FeedbackButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button
        size="lg"
        variant="outline"
        color="white"
        borderColor="whiteAlpha.600"
        _hover={{ bg: "whiteAlpha.200" }}
        onClick={() => setOpen(true)}
      >
        Give feedback
      </Button>
      {open && <FeedbackModal onClose={() => setOpen(false)} />}
    </>
  );
}
