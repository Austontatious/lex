import React, { useState } from "react";
import {
  Box,
  Button,
  Input,
  InputGroup,
  InputRightElement,
  Text,
  VStack,
} from "@chakra-ui/react";

interface TaskManagerProps {
  onSend: (payload: { prompt: string; task_type?: string }) => void;
  response: string;
}

const TaskManager: React.FC<TaskManagerProps> = ({ onSend, response }) => {
  const [prompt, setPrompt] = useState("");

  const handleSend = () => {
    if (prompt.trim() !== "") {
      onSend({ prompt }); // ✅ correct key
      setPrompt("");
    }
  };

  return (
    <VStack spacing={4} align="stretch" mt={6}>
      <Box>
        <InputGroup>
          <Input
            placeholder="Type your command..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
            size="lg"
            bg="whiteAlpha.100"
            color="white"
          />
          <InputRightElement width="6rem">
            <Button size="lg" onClick={handleSend} colorScheme="teal" minH={11} minW={11}>
              Send
            </Button>
          </InputRightElement>
        </InputGroup>
      </Box>
      <Box>
        <Text fontSize="sm" color="gray.400">
          Response:
        </Text>
        <Text fontSize="md" color="white">
          {response}
        </Text>
      </Box>
    </VStack>
  );
};

export default TaskManager;
